# Philosophie de transformation : Fortran → GPU → JAX

## Pourquoi cette séquence ?

```
Fortran (monolithique)
    │
    ▼  [extractor]
Fortran MODULAIRE (subroutines avec INTENT explicites)
    │
    ▼  [pure_elemental]
Fortran PUR (PURE/ELEMENTAL — fonctions sans effets de bord)
    │
    ▼  [openacc]
Fortran GPU (OpenACC !$acc parallel loop — exécution sur A100)
    │
    ▼  [cython_wrapper]
Python/Cython (interface NumPy memoryviews — appelable depuis Python)
    │
    ▼  [Phase 2, future]
JAX (jit/vmap — différentiable, fusionnable avec ML)
```

Chaque étape **active** l'étape suivante. Ce n'est pas une séquence arbitraire.

---

## Étape 1 — Extraction : monolithique → modulaire

**Problème** : les codes scientifiques comme `seismic_CPML_2D` sont des `PROGRAM` monolithiques.
Les boucles FD sont inline dans le programme principal, sans subroutines ni INTENT.

**Ce que fait l'agent** : il identifie les 4 boucles 2D (update_stress_xx_yy, update_stress_xy,
update_velocity_x, update_velocity_y) et les extrait dans un `MODULE Fortran` avec des arguments
explicitement typés (`INTENT(IN)`, `INTENT(INOUT)`).

**Pourquoi c'est nécessaire** :
- Sans INTENT explicites, on ne peut pas savoir quels tableaux sont en lecture seule
  (candidates `copyin`) vs modifiés in-place (candidates `copy`)
- Sans subroutines séparées, OpenACC ne peut pas ajouter `!$acc parallel loop` sur les bonnes boucles
- Sans MODULE, Cython ne peut pas générer de `cdef extern` propre

---

## Étape 2 — PURE/ELEMENTAL : effets de bord → fonctions pures

**PURE** : une subroutine sans I/O, sans SAVE, sans COMMON, sans modification de variables globales.
Tous les arguments ont un INTENT explicite.

**ELEMENTAL** : une fonction PURE qui opère sur un scalaire mais peut être appliquée element-wise
à un tableau. Équivalent de `jax.vmap` sur un axe.

**Pourquoi c'est crucial pour le GPU** :

| Propriété | Importance GPU | Importance JAX |
|-----------|---------------|----------------|
| Pas d'I/O | Les I/O ne peuvent pas s'exécuter sur device | idem |
| Pas de SAVE | Pas d'état caché entre appels → threads indépendants | Requis pour jit |
| INTENT explicite | Determine `copyin` vs `copy` pour OpenACC data | Determine les arguments JAX |
| Déterminisme | Résultat identique quel que soit l'ordre d'exécution des threads | Requis pour vmap |

**Exemples concrets** :
```fortran
! Avant : subroutine avec état implicite
subroutine update_stress(vx, vy, sigma_xx, ...)
  ! sigma_xx est modifiée mais Fortran ne le dit pas formellement
  sigma_xx(i,j) = sigma_xx(i,j) + ...
end subroutine

! Après PURE avec INTENT explicite
PURE subroutine update_stress(vx, vy, sigma_xx, ..., NX, NY, DELTAT)
  integer,          intent(in)    :: NX, NY
  double precision, intent(in)    :: vx(NX,NY), vy(NX,NY), DELTAT
  double precision, intent(inout) :: sigma_xx(NX,NY)   ! modifiée in-place
  ! ...
end subroutine
```

---

## Étape 3 — OpenACC : CPU séquentiel → GPU parallèle

**Pattern standard pour les stencils FD 2D** :

```fortran
PURE subroutine update_velocity_x(vx, sigma_xx, sigma_xy, rho, ..., NX, NY)
  ! Les scalaires temporaires doivent être private() (un par thread GPU)
  !$acc parallel loop collapse(2) private(value_dsigma_xx_dx, value_dsigma_xy_dy)
  do j = 2, NY
    do i = 2, NX
      value_dsigma_xx_dx = (sigma_xx(i,j) - sigma_xx(i-1,j)) / DELTAX
      value_dsigma_xy_dy = (sigma_xy(i,j) - sigma_xy(i,j-1)) / DELTAY
      vx(i,j) = vx(i,j) + (value_dsigma_xx_dx + value_dsigma_xy_dy) * DELTAT / rho(i,j)
    enddo
  enddo
  !$acc end parallel
end subroutine
```

**Dans le driver — data region autour du time loop** :

```fortran
! Copier les données UNE FOIS avant le time loop (2000 pas de temps)
!$acc data copyin(lambda,mu,rho,b_x,b_x_half,b_y,b_y_half,a_x,a_x_half,a_y,a_y_half,K_x,K_y,...) &
!$acc      copy(vx,vy,sigma_xx,sigma_yy,sigma_xy,memory_dvx_dx,memory_dvy_dx,...)

do it = 1, NSTEP
  call update_stress_xx_yy(...)
  call update_stress_xy(...)
  call update_velocity_x(...)
  call update_velocity_y(...)

  ! Avant I/O périodique : rapatrier vx,vy sur CPU pour affichage
  if (mod(it, IT_DISPLAY) == 0) then
    !$acc update host(vx, vy)
    velocnorm = maxval(sqrt(vx**2 + vy**2))
    print *, 'Time step #', it, 'velocnorm =', velocnorm
    call create_color_image(...)  ! reste sur CPU
  endif
enddo

!$acc end data
```

**Gain attendu** : sur A100, les 4 kernels FD 2D (NX=101, NY=641, NSTEP=2000) passent de
~10s (CPU) à ~0.1s (GPU) — facteur ×100 typique pour les stencils réguliers.

---

## Étape 4 — Cython : Fortran GPU → Python

Le wrapper Cython permet d'appeler les kernels GPU depuis Python/NumPy sans copie mémoire :

```python
# Côté Python (après compilation)
import numpy as np
import seismic_cpml_2d_gpu as gpu_module  # module Cython

vx = np.asfortranarray(np.zeros((NX, NY)))
vy = np.asfortranarray(np.zeros((NX, NY)))

# Appel direct du kernel Fortran GPU depuis Python
gpu_module.update_velocity_x(vx, sigma_xx, sigma_xy, rho, ...)
```

**Typed memoryviews** = pas de copie, accès direct au buffer NumPy.
**np.asfortranarray** = layout mémoire column-major (Fortran) = accès coalescé sur GPU.

---

## Phase 2 — JAX (future)

Une fois que les subroutines sont PURE avec INTENT explicites, la traduction JAX est directe :

| Fortran PURE | JAX équivalent |
|-------------|---------------|
| `PURE subroutine f(a, b, c_out)` | `@jax.jit def f(a, b) -> c` |
| `INTENT(IN)` | Argument JAX (immutable) |
| `INTENT(INOUT)` | Retourné par la fonction |
| `do i,j` (indépendants) | `jax.vmap` ou vectorisation implicite |
| `do it` (time loop avec état) | `jax.lax.scan` |
| `ELEMENTAL` | `jax.vmap(f, in_axes=0)` |

**Clé** : les subroutines PURE sont des **fonctions mathématiques pures** — c'est exactement
ce que JAX compile en XLA. La traduction Fortran→JAX pour des fonctions PURE est mécanique.

---

## Guide pratique — Tester sur le GPU Azure A100

### 1. Obtenir l'IP de la VM GPU

```bash
az vm list-ip-addresses -g rg-total-seismic-agent -o table
# ou
az vm show -g rg-total-seismic-agent -n vm-gpu-a100 \
    --show-details --query publicIps -o tsv
```

### 2. Lancer le pipeline agent

```bash
uv run agent-gpu translate /path/to/seismic_CPML_2D_isotropic_second_order.f90
# Produit : output/Makefile, output/compile_gpu.sh, output/fortran_gpu/*.f90
```

### 3. Déployer et compiler sur l'A100

```bash
# Option A : script automatique
AZURE_GPU_HOST=<ip> bash scripts/test_gpu.sh

# Option B : vérifier l'env GPU d'abord, puis compiler
AZURE_GPU_HOST=<ip> bash scripts/test_gpu.sh --check
AZURE_GPU_HOST=<ip> bash scripts/test_gpu.sh

# Option C : compilation directe via l'agent (si AZURE_GPU_HOST défini dans .env)
echo "AZURE_GPU_HOST=<ip>" >> .env
uv run agent-gpu translate /path/to/kernel.f90  # l'agent SSH et compile automatiquement
```

### 4. SSH direct pour debugger

```bash
ssh azureuser@<ip>
cd ~/seismic_gpu

# Vérifier le GPU
nvidia-smi

# Compiler manuellement
bash compile_gpu.sh

# Lancer la simulation GPU
./seismic_cpml_2d_isotropic_second_order_gpu

# Profiler l'exécution GPU
nsys profile ./seismic_cpml_2d_isotropic_second_order_gpu
```

### 5. Pour Pangea (HPC TotalEnergies)

Même logique, adaptez les variables :
```bash
AZURE_GPU_HOST=<pangea-node> AZURE_GPU_USER=<login_te> bash scripts/test_gpu.sh

# Sur Pangea, charger les modules HPC d'abord :
module load nvhpc/24.1   # NVIDIA HPC SDK (inclut nvfortran)
bash compile_gpu.sh
```

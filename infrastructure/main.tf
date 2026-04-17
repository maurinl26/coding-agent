terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Configuration du compte Azure
provider "azurerm" {
  features {}
}

# -------------------------------------------------------------
# 1. Groupe de ressources & Réseau Sécurisé
# -------------------------------------------------------------
resource "azurerm_resource_group" "rg" {
  name     = "rg-total-seismic-agent"
  location = "France Central" # Région typique TotalEnergies
}

resource "azurerm_virtual_network" "vnet" {
  name                = "vnet-seismic"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_subnet" "subnet_compute" {
  name                 = "snet-compute"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

# -------------------------------------------------------------
# 2. Instance d'Orchestration (Serveur LangGraph / FastMCP)
# -------------------------------------------------------------
resource "azurerm_network_interface" "nic_orch" {
  name                = "nic-orchestrator"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.subnet_compute.id
    private_ip_address_allocation = "Dynamic"
  }
}

resource "azurerm_linux_virtual_machine" "vm_orch" {
  name                = "vm-orchestrator-d8"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  size                = "Standard_D8s_v5" # CPU rapide, assez de RAM pour Loki-IFS
  admin_username      = "azureuser"
  
  network_interface_ids = [azurerm_network_interface.nic_orch.id]

  admin_ssh_key {
    username   = "azureuser"
    public_key = file("~/.ssh/id_rsa.pub")
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }
}

# -------------------------------------------------------------
# 3. Instance de Performance (Calcul JAX & Surrogate FNO)
# -------------------------------------------------------------
resource "azurerm_network_interface" "nic_gpu" {
  name                = "nic-gpu"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.subnet_compute.id
    private_ip_address_allocation = "Dynamic"
  }
}

resource "azurerm_linux_virtual_machine" "vm_gpu" {
  name                = "vm-gpu-a100"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  size                = "Standard_NC24ads_A100_v4" # 1x Nvidia A100 80GB (Ampere)
  admin_username      = "azureuser"
  
  network_interface_ids = [azurerm_network_interface.nic_gpu.id]

  admin_ssh_key {
    username   = "azureuser"
    public_key = file("~/.ssh/id_rsa.pub")
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 2048 # Stockage large nécessaire pour les tenseurs de gradient
  }

  # Image Optimisée DataScience : Contient déjà CUDA, les drivers Nvidia et Python
  source_image_reference {
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-2204"
    sku       = "2204-gen2"
    version   = "latest"
  }
}

program solver
  implicit none
  integer :: i, j
  real :: a(100,100), b(100,100), c(100,100)

  do i = 1, 100
    do j = 1, 100
      a(i,j) = i + j
      b(i,j) = i - j
      c(i,j) = 0.0
    end do
  end do

  print *, "Computation done"
end program solver

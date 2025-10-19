-- Proof that (a + b)^2 = a^2 + 2*a*b + b^2
theorem square_binomial (a b : Nat) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  simp only [Nat.pow_two]
  rw [Nat.add_mul, Nat.mul_add, Nat.mul_add]
  rw [Nat.mul_comm b a]
  rw [Nat.add_assoc (a * a)]
  rw [← Nat.add_assoc (a * b)]
  rw [← Nat.two_mul]
  rw [Nat.mul_assoc, Nat.add_assoc]

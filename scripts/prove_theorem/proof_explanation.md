# Proof of the Binomial Square Theorem

## Theorem Statement

For any natural numbers $a$ and $b$:

$$(a + b)^2 = a^2 + 2ab + b^2$$

## Proof

**Step 1:** Expand the square using the definition $x^2 = x \cdot x$

$$(a + b)^2 = (a + b)(a + b)$$

**Step 2:** Distribute the first term $(a + b)$ across the second term

$$= a(a + b) + b(a + b)$$

**Step 3:** Distribute again for each product

$$= a \cdot a + a \cdot b + b \cdot a + b \cdot b$$

**Step 4:** Use commutativity of multiplication ($b \cdot a = a \cdot b$)

$$= a \cdot a + a \cdot b + a \cdot b + b \cdot b$$

**Step 5:** Combine like terms ($a \cdot b + a \cdot b = 2 \cdot a \cdot b$)

$$= a^2 + 2ab + b^2$$

## Conclusion

Therefore, we have proven that $(a + b)^2 = a^2 + 2ab + b^2$ for all natural numbers $a$ and $b$. ∎

---

## Lean 4 Implementation

The formal proof in Lean uses the following tactics:
- `simp only [Nat.pow_two]` - Simplifies $x^2$ to $x \cdot x$
- `rw [Nat.add_mul, Nat.mul_add]` - Distribution laws
- `rw [Nat.mul_comm]` - Commutativity of multiplication
- `rw [← Nat.two_mul]` - Combines $x + x$ into $2x$
- `rw [Nat.add_assoc, Nat.mul_assoc]` - Associativity cleanup

The Lean proof verifier confirms this proof is logically sound and complete.

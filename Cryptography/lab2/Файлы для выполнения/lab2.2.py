from sympy import factorint

def main():
    A = 90040677735870852098413730802847928487482667469019156467295762242747218104249
    factors = factorint(A)

    print("Простые множители числа A:")
    for prime, power in factors.items():
        print(f"{prime}^{power}")

if __name__ == "__main__":
    main()

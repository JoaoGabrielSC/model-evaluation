def u(p, i, n):
    num = i * (1 + i) ** n
    den = (1 + i) ** n - 1
    return p * (num / den)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if len(args) != 3:
        print("Usage: python util.py <present value> <tax rate> <periodo>")
        print("Example: python util.py 1000 0.05 10")
        sys.exit(1)
    p = float(args[0])
    i = float(args[1])
    n = int(args[2])
    result = u(p, i, n)
    print(f"Future value: {result}")

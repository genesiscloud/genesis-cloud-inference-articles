
import sys

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 tab_perf.py <input_csv_path>")  

    input_path = sys.argv[1]

    min_col_width = 12
    margin = 4

    lines = []
    with open(input_path, "r") as fp:
        for line in fp:
            line = line.strip()
            lines.append(line)

    num_cols = len(lines[0].split(";"))
    col_widths = [min_col_width] * num_cols
    val_widths = [0] * num_cols

    for lno, line in enumerate(lines):
        fields = line.split(";")
        assert len(fields) == num_cols
        for col in range(num_cols):
            width = len(fields[col])
            if width > col_widths[col]:
                col_widths[col] = width
            if lno != 0 and width > val_widths[col]:
                val_widths[col] = width

    for lno, line in enumerate(lines):
        output = ""
        fields = line.split(";")
        for col in range(num_cols):
            field = fields[col]
            space = col_widths[col] - len(field)
            if col == 0:
                output += field          
                output += " " * space
            else:
                if lno == 0:
                    rpad = space // 2
                else:
                    rpad = (col_widths[col] - val_widths[col]) // 2
                lpad = space - rpad
                output += " " * (margin + lpad)
                output += field          
                output += " " * rpad
        print(output)
        if lno == 0:
            tab_width = sum(col_widths) + margin * (num_cols - 1)
            output = "-" * tab_width
            print(output)

main()



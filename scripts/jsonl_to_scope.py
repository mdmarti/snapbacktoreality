import json, sys, pathlib
import scope, elements

src, dst = sys.argv[1], elements.Path(sys.argv[2])
w = scope.Writer(dst)
for line in open(src):
    row = json.loads(line)
    step = row.pop('step')
    w.add(step, {k: float(v) for k, v in row.items()})
w.flush()

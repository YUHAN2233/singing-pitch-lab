# The application provides its own source-built LGPL ffmpeg executable.
# Override the contributed hook so the wheel's unrelated GPL binary is excluded.
datas = []
binaries = []

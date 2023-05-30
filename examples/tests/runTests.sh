#! /usr/bin/env nix-shell
#! nix-shell -i bash ../../compiler/shell.nix

verbose="$1"

if [ "$verbose" == "1" ]; then
    cmd="diff -y"
else
    cmd="diff"
fi
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# https://stackoverflow.com/questions/4638874/how-to-loop-through-a-directory-recursively-to-delete-files-with-certain-extensi
for f in $(find "$SCRIPT_DIR" -name '*.txt'); do
    output="$($cmd "$f" <(python3 "$SCRIPT_DIR/../../compiler/main.py" "../$(realpath --relative-to "." "$f")"))"
    if [ ! -z "$output" ]; then
	echo "Test $f failed:"
	echo "$output"
    fi
done

echo "Done"

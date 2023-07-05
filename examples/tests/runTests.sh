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
    fOutput="$f"
    f="$SCRIPT_DIR/../$(realpath --relative-to "$SCRIPT_DIR" "$f")"
    output="$(diff "$fOutput" <(python3 "$SCRIPT_DIR/../../compiler/main.py" "$f" 2>&1))"
    if [ ! -z "$output" ]; then
	echo "Test $fOutput failed:"
	if [ "$verbose" == "1" ]; then
	    diff -y "$fOutput" <(python3 "$SCRIPT_DIR/../../compiler/main.py" "$f" 2>&1)
	else
	    echo "$output"
	fi
    fi
done

echo "Done"

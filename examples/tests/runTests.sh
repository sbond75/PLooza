#! /usr/bin/env nix-shell
#! nix-shell -i bash ../../compiler/shell.nix

verbose="$1"
# Environment variables that can be provided: `debug=1` to debug this script.

if [ "$verbose" == "1" ]; then
    cmd="diff -y"
else
    cmd="diff"
fi
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# https://stackoverflow.com/questions/4638874/how-to-loop-through-a-directory-recursively-to-delete-files-with-certain-extensi
for f in $(find "$SCRIPT_DIR" -name '*.txt' ! -name '*.stdin.txt'); do
    fOutput="$f"
    f="$SCRIPT_DIR/../$(realpath --relative-to "$SCRIPT_DIR" "$f")"

    path=$(dirname -- "$fOutput")
    filename=$(basename -- "$fOutput")
    extension="${filename##*.}"
    filename="${filename%.*}"
    if [ "$debug" == "1" ]; then
	echo "$filename"
    fi
    stdinFile="$path/${filename}.stdin.txt"
    # if [ "$debug" == "1" ]; then
    # 	if [ -e "$stdinFile" ]; then
    # 	    echo "stdin file exists: $stdinFile"
    # 	else
    # 	    echo "stdin file doesn't exist: $stdinFile"
    # 	fi
    # fi
    
    if [ -e "$stdinFile" ]; then
	# Use stdin from this file
	output="$(diff "$fOutput" <(cat "$stdinFile" | python3 "$SCRIPT_DIR/../../compiler/main.py" "$f" 2>&1))"
    else
	output="$(diff "$fOutput" <(python3 "$SCRIPT_DIR/../../compiler/main.py" "$f" 2>&1))"
    fi
    
    if [ ! -z "$output" ]; then
	echo "Test $fOutput failed:"
	if [ "$verbose" == "1" ]; then
	    if [ -e "$stdinFile" ]; then
		# Use stdin from this file
		diff -y "$fOutput" <(cat "$stdinFile" | python3 "$SCRIPT_DIR/../../compiler/main.py" "$f" 2>&1)
	    else
		diff -y "$fOutput" <(python3 "$SCRIPT_DIR/../../compiler/main.py" "$f" 2>&1)
	    fi
	else
	    echo "$output"
	fi
    fi
done

echo "Done"

#!/usr/bin/env bash
# vim: set noexpandtab tabstop=2:

max_length=200
batch_size=32
num_epochs=10
learning_rate=0.001
outdir='model'

usage="Usage:\n\t$0 <-i input_file> [-m max_length_of_sequence] [-b batch_size] [-n number_of_epochs] [-l learning_rate] [-o outdir_of_model]"

declare -g -A __sourced__files__
if [[ ! -v __sourced__files__[$BASH_SOURCE] || $__force__source__ ]]; then
	__sourced__files__[$BASH_SOURCE]=$(realpath "$BASH_SOURCE")
	path=$(dirname $(realpath "$BASH_SOURCE"))
	function lstm_text_classifier {
		while true; do
			case $1 in
				-h|--help)
					echo -e $usage
					return
					;;
				-i)
					infile="$2"
					shift
					shift
					;;
				-m)
		      max_length="$2"
					shift
					shift
					;;
				-b)
          batch_size="$2"
					shift
					shift
					;;
				-n)
				  num_epochs="$2"
					shift
					shift
					;;
				-l)
				  learning_rate="$2"
					shift
					shift
					;;
				-o)
				  outdir="$2"
					shift
					shift
					;;
				-*)
				  echo "$FUNCNAME:ERROR: Bad option '$1'." >&2
					echo -e $usage
					return -1
					;;
				*)
				  break
					;;
			esac
		done

    if [ ! -d $outdir ];then
      mkdir -p $outdir
		fi

    "$path"/main.py $infile $max_length $batch_size $num_epochs $learning_rate $outdir
	}

	if ! { ( return ) } 2>/dev/null; then
		set -e
		lstm_text_classifier "$@" || exit
	fi
fi


#!/bin/bash

COPYRIGHT_LINE_RE="Copyright (c) .* The University of Tennessee"
SCRIPTNAME="copyright-check.sh"

date_last_git_mod() {
     git log -1 --pretty="format:%cs" $1 | awk -v FS="-" '{print $1}'
}

has_copyright() {
    grep -q "$COPYRIGHT_LINE_RE" $1
}

copyright_end_year() {
    if has_copyright $1
    then
        grep "$COPYRIGHT_LINE_RE" $1 | awk -v FS="[ -]+" '$6=="The" {print $5; nextfile} {print $6}'
    else
        echo "File '$1' does not seem to have a copyright entry"  >> /dev/stderr
        return 1
    fi
    return 0
}

usage()
{
  echo "Usage: $SCRIPTNAME [-f | --find pattern] [-o | --old] [-m | --missing] [filename(s)]"
  exit 2
}

MISSING=0
OLD=0
VERBOSE=0

PARSED_ARGUMENTS=$(/usr/bin/getopt -a -n $SCRIPTNAME -o f:omv -l find:,old,missing,verbose -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi
PARSED_ARGUMENTS=$(echo $PARSED_ARGUMENTS | sed -e 's/--//')

files=""
eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -f | --find)    for f in $(find . -name $2); do files="$files $f"; done; shift 2  ;;
    -m | --missing) MISSING=1                    ; shift    ;;
    -o | --old)     OLD=1                        ; shift    ;;
    -v | --verbose) VERBOSE=1                    ; shift    ;;
    **)                             files="$files $1"; [ -n "$1" ] && shift || break ;;
  esac
done
for m in ${@}
do
   files="$files $1"
done

for m in ${files}
do
    if has_copyright $m
    then
        if [ $OLD -eq 1 ]; then
            gly=$(date_last_git_mod $m)
            cey=$(copyright_end_year $m)
            if [ "$cey" != "$gly" ]; then
                if [ $VERBOSE -eq 1 ]; then
                    echo "O $m (last update: $gly current copyright: $cey"
                else
                    echo "$m"
                fi
            else
                if [ $VERBOSE -eq 1 ]; then
                    echo "  $m"
                fi
            fi
        fi
    else
        if [ $MISSING -eq 1 ]; then
            if [ $VERBOSE -eq 1 ]; then
                echo -n "M "
            fi
            echo "$m"
        fi
    fi
done

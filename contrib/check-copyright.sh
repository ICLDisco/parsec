#!/bin/bash
COPYRIGHT_LINE_C="Copyright (c)"
COPYRIGHT_LINE_U="The University of Tennessee"
COPYRIGHT_LINE_RE="$COPYRIGHT_LINE_C .* $COPYRIGHT_LINE_U"
SCRIPTNAME="copyright-check.sh"

list_all_git_files() {
  git ls-files
}

date_last_git_mod() {
     git log -1 --pretty="format:%cs" $1 | awk -v FS="-" '{print $1}'
}

date_first_git_mod() {
    git log --pretty="format:%cs" $1 | awk -v FS="-" 'END {print $1}'
}

has_copyright() {
    grep -qI "$COPYRIGHT_LINE_RE" $1
}

get_copyright() {
    if has_copyright $1
    then
        local cline=$(grep "$COPYRIGHT_LINE_RE" $1)
        cline="${cline##*${COPYRIGHT_LINE_C} }"
        cline="${cline%% ${COPYRIGHT_LINE_U}*}"
        echo "$cline"
    else
        echo "File '$1' does not seem to have a copyright entry"  >> /dev/stderr
        return 1
    fi
    return 0
}

copyright_start_year() {
    echo ${1%%-*}
}

copyright_end_year() {
  echo ${1##*-}
}

usage()
{
  echo "Usage: $SCRIPTNAME [-f | --find pattern] [-o | --old] [-u | --update] [-m | --missing] [filename(s)]"
  exit 2
}

MISSING=0
OLD=0
UPDATE=0
VERBOSE=0

PARSED_ARGUMENTS=$(/usr/bin/getopt -a -n $SCRIPTNAME -o f:oumv -l find:,old,update,missing,verbose -- "$@")
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
    -u | --update)  OLD=1; UPDATE=1              ; shift    ;;
    -v | --verbose) VERBOSE=1                    ; shift    ;;
    **)             [ -n "$1" ] && files="$files $1" && shift || break ;;
  esac
done
for m in ${@}
do
   files="$files $1"
done

if [ -z "${files}" ]
then
  files=$(list_all_git_files)
fi


for m in ${files}
do
    if has_copyright $m
    then
        if [ $OLD -eq 1 ]; then
            cpr=$(get_copyright $m)
            csy=$(copyright_start_year "$cpr")
            cey=$(copyright_end_year "$cpr")
            gly=$(date_last_git_mod $m)
            if [ "$cey" != "$gly" ]; then
                if [ $VERBOSE -eq 1 ]; then
                    printf "U %-80s %-9s -> %-9s\n" $m $cpr "$csy-$gly"
                else
                    echo "$m"
                fi
                if [ $UPDATE -eq 1 ]; then
                    sed -i "s/$COPYRIGHT_LINE_C $cpr $COPYRIGHT_LINE_U/$COPYRIGHT_LINE_C $csy-$gly $COPYRIGHT_LINE_U/" $m
                fi
            else
                if [ $VERBOSE -eq 1 ]; then
                    echo "= $m"
                fi
            fi
        fi
    else
        if [ $MISSING -eq 1 ]; then
            if [ $VERBOSE -eq 1 ]; then
                gly=$(date_last_git_mod $m)
                gey=$(date_first_git_mod $m)
                printf "A %-80s %-9s -> %-9s\n" $m "" "$gey-$gly"
            else
                echo "$m"
            fi
        fi
    fi
done

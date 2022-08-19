#!/usr/bin/env bash

# example usage for parse_option.sh in file.sh
###
### $ cat file.sh
### param=
### help_message="define help function"
### . ./parse_options.sh
### echo "param: $param"
###
### $ file.sh --param value
### param: value
###

# check for valid --config options
for (( argPos=1; argPos<=$#; argPos++ )); do
  if [ "${!argPos}" = '--config' ]; then
    argPosPlus=$((argPos+1))
    config=${!argPosPlus}
    [ ! -r "$config" ] && echo "--config option is restricted for [FilePath] only" && exit 1;
    . "$config";
  fi
done

sal=bas

while true; do
  [ -z "${1:-}" ] || [ -z "${2:-}" ] && break;

  case "$1" in
  --help|-h) [ -z "$help_message" ] && echo "No help options found." || printf "%s\n" "$help_message" 1>&2;
    exit 0 ;;
  --*=*) echo "$0: options to script must be of the form --option value, got $1"
    exit 1 ;;
  --*) name=$( echo "$1" | sed s/^--// | sed s/-/_/ )
    eval '[ -z "${'$name'+is_defined}" ]' && echo "$0: invalid option $1" && exit 1;

    old_val="$(eval echo \$'$name')"

    if [ "$old_val" = "true" ] || [ "$old_val" = "false" ]; then
      is_boolean=true;
    else
      is_boolean=false;
    fi

    #
    eval "$name"=\""$2"\";

    if "$is_boolean" && [[ "$name" != "false" || "$name" != "true" ]]; then
      echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
      exit 1;
    fi
    shift 2;
    ;;
  *) break;
  esac
done

true;

#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
mkdir -p $SCRIPTPATH/../badges

score=$(poetry run pylint utensil | grep ^Your | tail -1 | sed 's|Your code has been rated at \([\.0-9]\+\)/10.*|\1|g')
if (( $(echo "$score >= 10" | bc -l) )) ; then
    color="brightgreen"
elif (( $(echo "$score >= 9" | bc -l) )) ; then
    color="green"
elif (( $(echo "$score >= 8" | bc -l) )) ; then
    color="yellowgreen"
elif (( $(echo "$score >= 7" | bc -l) )) ; then
    color="yellow"
elif (( $(echo "$score >= 6" | bc -l) )) ; then
    color="orange"
else
    color="red"
fi
wget -O $SCRIPTPATH/../badges/pylint.svg "https://img.shields.io/badge/pylint-${score}-${color}"

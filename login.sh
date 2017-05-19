#!/usr/bin/env sh

curl -d 'DDDDD=2016010052&upass=217217&R1=0' '10.3.8.211' -o output.html && rm output.html && echo 'Login Successfully!'

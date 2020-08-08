#!/usr/bin/env bash
kill -9 $(ps aux | grep $1 | awk '{print $2}')
#!/bin/bash

# IDK

grep -vP '[\p{Han}]' \
  | grep -vP '[\p{Hiragana}]' \
  | grep -vP '[\p{Katakana}]' \
  | grep -vP '[\p{Hebrew}]' \
  | grep -vP '[\p{Cyrillic}]' \
  | grep -vP '[\p{Arabic}]' 

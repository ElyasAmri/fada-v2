#!/bin/bash
ssh -o ConnectTimeout=10 fada-1 'rm -rf ~/.cache/pip && rm -rf ~/.cache/huggingface && echo cleaned && df -h /'

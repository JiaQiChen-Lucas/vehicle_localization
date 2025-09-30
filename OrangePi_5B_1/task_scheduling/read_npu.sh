#!/bin/bash
echo "Running ..."
while true; do
    cat /sys/kernel/debug/rknpu/load > /tmp/npu_load
    sleep 1
done

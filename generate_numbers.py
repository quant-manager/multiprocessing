#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: James Johnson
# @file_name: generate_numbers.py
#
# Copyright 2024 James James Johnson. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
References:
-----------
# https://stackoverflow.com/questions/11853551/python-multiple-users-append-to-the-same-file-at-the-same-time
"""


import numpy as np
import time
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from filelock import FileLock
import platform
STR_PLATFORM_SYSTEM = platform.system()
# https://www.adventuresinmachinelearning.com/ensuring-data-integrity-implementing-file-locking-in-python/
if STR_PLATFORM_SYSTEM == 'Linux' :
    # https://stackoverflow.com/questions/11853551/python-multiple-users-append-to-the-same-file-at-the-same-time
    # https://docs.python.org/3/library/fcntl.html
    import fcntl
#if STR_PLATFORM_SYSTEM == 'Windows' :
#    # https://stackoverflow.com/questions/30440559/how-to-perform-file-locking-on-windows-without-installing-a-new-package
#    # https://docs.python.org/3/library/msvcrt.html#module-msvcrt
#    import msvcrt


INT_NANOSECONDS_PER_MILLISECOND = 1_000_000
INT_INTEGER_MINIMUM_VALUE = np.iinfo(np.uint16).min
INT_INTEGER_MAXIMUM_VALUE = np.iinfo(np.int16).max


def generate_data_batch(int_batch_size = 1, int_batch_seed = None) :
    random_generator = np.random.Generator(np.random.PCG64(
        seed = int_batch_seed))
    return random_generator.integers(
        INT_INTEGER_MINIMUM_VALUE,
        INT_INTEGER_MAXIMUM_VALUE + 1,
        size = int_batch_size).tolist()


def pack_data_batch(lst_batch_data, int_max_num_digits) :
    return "\n".join([str(i).zfill(int_max_num_digits) for i in lst_batch_data])


def process_data_batch(
        str_file_name, int_max_num_digits,
        int_batch_size, int_batch_seed = None,):
    lst_batch_data = generate_data_batch(
        int_batch_size = int_batch_size,
        int_batch_seed = int_batch_seed,)
    bytes_packed_data_batch = (pack_data_batch(
        lst_batch_data = lst_batch_data,
        int_max_num_digits = int_max_num_digits,) + "\n").encode("utf-8")
    if STR_PLATFORM_SYSTEM == 'Linux' :
        with open(str_file_name, 'a+b') as file_handle :
            fcntl.flock(file_handle, fcntl.LOCK_EX)
            file_handle.write(bytes_packed_data_batch)
            fcntl.flock(file_handle, fcntl.LOCK_UN)
    elif STR_PLATFORM_SYSTEM == 'Windows' :
        lock = FileLock(str_file_name + ".lock")
        with lock:
            with open(str_file_name, 'a+b') as file_handle :
                try :
                    file_handle.write(bytes_packed_data_batch)
                except Exception as e :
                    print(str(e))


def obtain_inputs() :

    int_max_num_parallel_processes = -1
    while int_max_num_parallel_processes <= 0 :
        # 5, 10
        str_max_num_parallel_processes = input(
            "Enter the maximum number of parallel processes ( > 0): ")
        if str_max_num_parallel_processes.isdigit() :
            int_max_num_parallel_processes = int(str_max_num_parallel_processes)

    int_ttl_num_ints_per_file = -1
    while int_ttl_num_ints_per_file <= 0 :
        # 100, 1_000_000_000
        str_ttl_num_ints_per_file = input(
            "Enter the total number of integers per file ( > 0): ")
        if str_ttl_num_ints_per_file.isdigit() :
            int_ttl_num_ints_per_file = int(str_ttl_num_ints_per_file)

    int_max_num_ints_per_batch = -1
    while int_max_num_ints_per_batch <= 0 :
        # 10, 1_000_000
        str_max_num_ints_per_batch = input(
            "Enter the maximum number of integers per batch ( > 0): ")
        if str_max_num_ints_per_batch.isdigit() :
            int_max_num_ints_per_batch = int(str_max_num_ints_per_batch)

    str_left_file_name = input("Enter the left input file name: ") # "file1.txt"

    str_right_file_name = input("Enter the right input file name: ") # "file2.txt"

    int_left_file_random_seed = -1
    while int_left_file_random_seed < 0 :
        # 12345
        str_left_file_random_seed = input(
            "Enter the left file random seed ( >= 0): ")
        if str_left_file_random_seed.isdigit() :
            int_left_file_random_seed = int(str_left_file_random_seed)

    int_right_file_random_seed = -1
    while int_right_file_random_seed < 0 :
        # 54321
        str_right_file_random_seed = input(
            "Enter the right file random seed ( >= 0): ")
        if str_right_file_random_seed.isdigit() :
            int_right_file_random_seed = int(str_right_file_random_seed)

    return (int_max_num_parallel_processes, int_ttl_num_ints_per_file,
            int_max_num_ints_per_batch,
            str_left_file_name, str_right_file_name,
            int_left_file_random_seed, int_right_file_random_seed,)


def generate_file(
        int_max_num_parallel_processes,
        str_file_name, int_ttl_num_ints, int_max_num_ints_per_batch,
        int_random_seed,) :
    try :
        os.remove(str_file_name)
    except OSError :
        pass
    try :
        int_max_num_digits = len(str(INT_INTEGER_MAXIMUM_VALUE))
        int_start_timestamp_nanoseconds = time.monotonic_ns()
        with ProcessPoolExecutor(
                max_workers = int_max_num_parallel_processes) as \
            process_pool_executor :
            int_batch_index = 0
            int_batch_start_offset = 0
            int_batch_size = min(
                int_max_num_ints_per_batch,
                (int_ttl_num_ints - int_batch_start_offset))
            lst_futures = []
            while int_batch_start_offset < int_ttl_num_ints :
                int_batch_seed = \
                    int_random_seed % (int_batch_index + 1) + \
                    (int_batch_index + 1) % int_random_seed + \
                    int_batch_index + int_random_seed
                lst_futures.append(process_pool_executor.submit(
                    process_data_batch, str_file_name,
                    int_max_num_digits,
                    int_batch_size, int_batch_seed,))
                int_batch_index += 1
                int_batch_start_offset += int_batch_size
                int_batch_size = min(
                    int_max_num_ints_per_batch,
                    (int_ttl_num_ints - int_batch_start_offset))
            wait(lst_futures)
        int_end_timestamp_nanoseconds = time.monotonic_ns()
        int_script_runtime_nanoseconds = int_end_timestamp_nanoseconds - \
            int_start_timestamp_nanoseconds
        int_script_runtime_milliseconds = int(round(float(
            int_script_runtime_nanoseconds) /
            INT_NANOSECONDS_PER_MILLISECOND))
        print('The file "' + str_file_name + '" was generated within ' +
              str(int_script_runtime_milliseconds) + " milliseconds.")
    except Exception as e:
        print(str(e))


def main() :

    # Requirement: the maximum integer value is known, its number of digits is
    # counted (N), and each integer is padded with leading zeros to assure
    # fixed length (of the maximum value) for all values.

    if True :
        (int_max_num_parallel_processes, int_ttl_num_ints_per_file,
         int_max_num_ints_per_batch,
         str_left_file_name, str_right_file_name,
         int_left_file_random_seed, int_right_file_random_seed,) = \
            obtain_inputs()

    if False :
        (int_max_num_parallel_processes, int_ttl_num_ints_per_file,
         int_max_num_ints_per_batch,
         str_left_file_name, str_right_file_name,
         int_left_file_random_seed, int_right_file_random_seed,) = \
        (5, 100, 10, "file1.txt", "file2.txt", 12345, 54321,)

    if False :
        (int_max_num_parallel_processes, int_ttl_num_ints_per_file,
         int_max_num_ints_per_batch,
         str_left_file_name, str_right_file_name,
         int_left_file_random_seed, int_right_file_random_seed,) = \
        (10, 1_000_000, 100_000, "file1.txt", "file2.txt", 12345, 54321,)

    if False :
        (int_max_num_parallel_processes, int_ttl_num_ints_per_file,
         int_max_num_ints_per_batch,
         str_left_file_name, str_right_file_name,
         int_left_file_random_seed, int_right_file_random_seed,) = \
        (10, 1_000_000_000, 1_000_000, "hugefile1.txt", "hugefile2.txt",
         12345, 54321,)

    generate_file(
        int_max_num_parallel_processes = int_max_num_parallel_processes,
        str_file_name = str_left_file_name,
        int_ttl_num_ints = int_ttl_num_ints_per_file,
        int_max_num_ints_per_batch = int_max_num_ints_per_batch,
        int_random_seed = int_left_file_random_seed,)

    generate_file(
        int_max_num_parallel_processes = int_max_num_parallel_processes,
        str_file_name = str_right_file_name,
        int_ttl_num_ints = int_ttl_num_ints_per_file,
        int_max_num_ints_per_batch = int_max_num_ints_per_batch,
        int_random_seed = int_right_file_random_seed,)


if __name__ == '__main__':
    main()

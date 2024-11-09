#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: James Johnson
# @file_name: add_numbers.py
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

1. https://docs.python.org/3/library/multiprocessing.html
2. https://www.tutorialspoint.com/concurrency_in_python/concurrency_in_python_processes_intercommunication.htm
3. https://stackoverflow.com/questions/9436757/how-to-use-a-multiprocessing-manager
"""


import numpy as np
import os
import time
import datetime
import threading
import multiprocessing
import sklearn.utils
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import platform
STR_PLATFORM_SYSTEM = platform.system()
# https://www.adventuresinmachinelearning.com/ensuring-data-integrity-implementing-file-locking-in-python/
if STR_PLATFORM_SYSTEM == 'Linux' :
    # https://stackoverflow.com/questions/11853551/python-multiple-users-append-to-the-same-file-at-the-same-time
    # https://docs.python.org/3/library/fcntl.html
    import fcntl


INT_NANOSECONDS_PER_MILLISECOND = 1_000_000
INT_INTEGER_MAXIMUM_INPUT_VALUE = np.iinfo(np.int16).max
INT_INTEGER_MAXIMUM_OUTPUT_VALUE = np.iinfo(np.uint16).max


# This function is fully agnostic of the "fn_job" interface!
def fn_worker_process(*args):

    # This daemon process. Daemon processes are abruptly stopped at shutdown.
    int_process_name = args[0]
    queue_parallel_jobs = args[1] # The queue of dictionaries for "Thread"s!!!
    multiproc_manager_namespace = args[2]
    multiproc_lock = args[3]
    while True:
        # No need for multiprocessing.Lock()'s acquire and release() for
        # accessing "queue_threaded_jobs" or decreasing its counter.
        dict_thread_params = queue_parallel_jobs.get(
            block = True, # blocking point and main parallel shut down point
            timeout = None, # no timeout
            )
        threaded_job = threading.Thread(
            target = dict_thread_params["target"],
            name = dict_thread_params["name"],
            args = (*(dict_thread_params["args"]),
                    multiproc_manager_namespace,
                    multiproc_lock),
            ) # dict_thread_params["args"])
        print("\nProcess '{:}' unblocked for Job '{}' at ".format(
            int_process_name, threaded_job.name) +
            datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            flush = True, end="")
        if threaded_job.name == "Exit":
            queue_parallel_jobs.task_done()
            break
        threaded_job.start() # Start a Thread (not a Process!!!)
        threaded_job.join() # another blocking point
        # Just decrease the counter, not job-specific, can be from any thread!
        queue_parallel_jobs.task_done()
        print("\nProcess '{:}' finished Job '{}' at ".format(
            int_process_name, threaded_job.name) +
            datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            flush = True, end = "")
    print("\nProcess '{:}' is exiting at ".format(int_process_name) +
          datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
          flush = True, end = "")


def fn_subjob2_to_read_batch_data(
        str_input_file_name,
        str_trace_file_name,
        int_netto_input_row_length_bytes,
        int_batch_slice_start,
        int_batch_slice_stop,) :

    # Run the job code with high I/O utilization
    #lst_int_batch_input_data = None
    np_arr_uint16_batch_input_data = None
    bytes_batch_input_data = None
    int_brutto_input_row_length_bytes = int_netto_input_row_length_bytes + len(b'\n')
    int_batch_slice_start_adj = max(
        0, int_batch_slice_start - int_brutto_input_row_length_bytes)
    int_batch_size_in_bytes = int_batch_slice_stop - int_batch_slice_start_adj
    try :
        with open(str_input_file_name, 'rb') as file_input_handle :
            # Read extra bytes before the requested starting position.
            file_input_handle.seek(int_batch_slice_start_adj, 0)
            bytes_batch_input_data = file_input_handle.read(
                int_batch_size_in_bytes)
    except :
        pass
    if bytes_batch_input_data is not None :
        int_batch_slice_start_fit = -1
        if int_batch_slice_start == 0 :
            int_batch_slice_start_fit = 0
        else :
            # Cut the head starting from the nearest (last) new line
            # (including this new line) to the left from int_batch_slice_start.
            int_batch_slice_start_fit = bytes_batch_input_data[
                : int_brutto_input_row_length_bytes].rfind(b'\n')
            if int_batch_slice_start_fit != -1 :
                int_batch_slice_start_fit += 1
        if int_batch_slice_start_fit != -1 and \
           int_batch_slice_start_fit < len(bytes_batch_input_data) :
            # Cut the tail of the buffer after the last new line.
            int_batch_slice_stop_fit = bytes_batch_input_data.rfind(b'\n')
            if int_batch_slice_stop_fit != -1 :
                int_batch_slice_stop_fit += 1
                if int_batch_slice_start_fit != int_batch_slice_stop_fit and \
                   int_batch_slice_stop_fit <= len(bytes_batch_input_data) :
                    # Use bytes_batch_input_data with adjusted boundaries:
                    try :
                        with open(str_trace_file_name, 'w+b') as file_trace_handle :
                            file_trace_handle.write(bytes_batch_input_data[
                                int_batch_slice_start_fit:int_batch_slice_stop_fit])
                    except :
                        pass
                    # Convert bytearray with "utf-8" string encoding of decimal
                    # integers into "np.uint16" type.
                    int_batch_size_uint16 = (
                        int_batch_slice_stop_fit - int_batch_slice_start_fit) // \
                            int_brutto_input_row_length_bytes
                    np_arr_uint16_batch_input_data = np.empty([
                        int_batch_size_uint16], dtype=np.uint16)
                    int_uint16_offset = 0
                    for int_byte_offset in range(int_batch_slice_start_fit,
                                   int_batch_slice_stop_fit,
                                   int_brutto_input_row_length_bytes) :
                        np_arr_uint16_batch_input_data[int_uint16_offset] = \
                            np.uint16(bytes_batch_input_data[
                                int_byte_offset : (
                                    int_byte_offset +
                                    int_netto_input_row_length_bytes)].decode(
                                        encoding='utf-8'))
                        int_uint16_offset += 1
    return np_arr_uint16_batch_input_data


def fn_subjob3_to_add_right_to_left(
        np_arr_uint16_batch_left_input_data,
        np_arr_uint16_batch_right_input_data,
        int_netto_output_row_length_bytes,) :

    # Run computations
    np_arr_uint16_batch_output_data = np_arr_uint16_batch_left_input_data + \
        np_arr_uint16_batch_right_input_data
    int_batch_output_data_checksum = int(0)
    for uint16_value in np_arr_uint16_batch_output_data :
        int_batch_output_data_checksum += int(uint16_value)

    # Convert "np.uint16" type into bytearray with "utf-8" string encoding of
    # decimal integers.
    int_batch_size_uint16 = len(np_arr_uint16_batch_output_data)
    int_brutto_output_row_length_bytes = int_netto_output_row_length_bytes + len(b'\n')
    int_bytearray_buffer_size = int_batch_size_uint16 * \
        int_brutto_output_row_length_bytes
    bytearray_batch_output_data = bytearray(int_bytearray_buffer_size)
    int_uint16_offset = 0
    byte_new_line = '\n'.encode("utf-8")
    for int_byte_offset in range(0, int_bytearray_buffer_size,
                                 int_brutto_output_row_length_bytes) :
        bytearray_batch_output_data[
            int_byte_offset:(int_byte_offset+int_netto_output_row_length_bytes)] = \
            str(np_arr_uint16_batch_output_data[int_uint16_offset]).zfill(
                int_netto_output_row_length_bytes).encode("utf-8")
        bytearray_batch_output_data[
            (int_byte_offset + int_netto_output_row_length_bytes) :
                (int_byte_offset + int_netto_output_row_length_bytes + 1)] = \
            byte_new_line
        int_uint16_offset += 1
    return (bytearray_batch_output_data, int_batch_output_data_checksum)


def fn_subjob4_to_save_results(
        bytearray_batch_output_data, str_output_file_name) :
    try :
        if STR_PLATFORM_SYSTEM == 'Linux' :
            with open(str_output_file_name, 'a+b') as file_handle :
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
                file_handle.write(bytearray_batch_output_data)
                fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
        elif STR_PLATFORM_SYSTEM == 'Windows' :
            lock = FileLock(str_output_file_name + ".lock")
            with lock:
                with open(str_output_file_name, 'a+b') as file_handle :
                    file_handle.write(bytearray_batch_output_data)
    except OSError :
        pass


def fn_job(*args) :

    # Partially unpackage job inputs
    int_job_index = args[0]

    tpl_job_inputs = args[1]
    str_left_input_file_name = tpl_job_inputs[0]
    str_right_input_file_name = tpl_job_inputs[1]
    str_trace_file_name_prefix = tpl_job_inputs[2]
    str_output_file_name = tpl_job_inputs[3]
    int_netto_input_row_length_bytes = tpl_job_inputs[4]
    int_netto_output_row_length_bytes = tpl_job_inputs[5]
    int_batch_slice_start = tpl_job_inputs[6]
    int_batch_slice_stop = tpl_job_inputs[7]

    print("\nJob '" + str(int_job_index) + "' is started for batch [" +
          str(int_batch_slice_start) + ", " + str(int_batch_slice_stop) + ")" +
          " at " + datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
          flush = True, end="")

    if False :
        # Read two input files sequentially in this worker process.
        # Sub-job #2.1
        np_arr_uint16_batch_left_input_data = fn_subjob2_to_read_batch_data(
            str_input_file_name = str_left_input_file_name,
            str_trace_file_name = str_trace_file_name_prefix + "_left_" + \
                str(int_job_index) + ".txt",
            int_netto_input_row_length_bytes = int_netto_input_row_length_bytes,
            int_batch_slice_start = int_batch_slice_start,
            int_batch_slice_stop = int_batch_slice_stop,)

        # Sub-job #2.2
        np_arr_uint16_batch_right_input_data = fn_subjob2_to_read_batch_data(
            str_input_file_name = str_right_input_file_name,
            str_trace_file_name = str_trace_file_name_prefix + "_right_" + \
                str(int_job_index) + ".txt",
            int_netto_input_row_length_bytes = int_netto_input_row_length_bytes,
            int_batch_slice_start = int_batch_slice_start,
            int_batch_slice_stop = int_batch_slice_stop,)
    else :
        # Read two input files concurrently in two worker threads in this
        # worker process.
        with ThreadPoolExecutor(max_workers = 2) as thread_pool_executor :
            lst_futures = []
            # Sub-job #2.1
            lst_futures.append(thread_pool_executor.submit(
                fn_subjob2_to_read_batch_data,
                str_input_file_name = str_left_input_file_name,
                str_trace_file_name = str_trace_file_name_prefix + "_left_" + \
                    str(int_job_index) + ".txt",
                int_netto_input_row_length_bytes = int_netto_input_row_length_bytes,
                int_batch_slice_start = int_batch_slice_start,
                int_batch_slice_stop = int_batch_slice_stop,))
            # Sub-job #2.2
            lst_futures.append(thread_pool_executor.submit(
                fn_subjob2_to_read_batch_data,
                str_input_file_name = str_right_input_file_name,
                str_trace_file_name = str_trace_file_name_prefix + "_right_" + \
                    str(int_job_index) + ".txt",
                int_netto_input_row_length_bytes = int_netto_input_row_length_bytes,
                int_batch_slice_start = int_batch_slice_start,
                int_batch_slice_stop = int_batch_slice_stop,))

            # Output lists may be flipped non-deterministically. But this is
            # acceptable, since addition is commutative.
            (np_arr_uint16_batch_left_input_data,
             np_arr_uint16_batch_right_input_data) = \
                (future.result() for future in as_completed(lst_futures))

    # Sub-job #3
    (bytearray_batch_output_data, int_batch_output_data_checksum) = \
        fn_subjob3_to_add_right_to_left(
            np_arr_uint16_batch_left_input_data = np_arr_uint16_batch_left_input_data,
            np_arr_uint16_batch_right_input_data = np_arr_uint16_batch_right_input_data,
            int_netto_output_row_length_bytes = int_netto_output_row_length_bytes,)
    del np_arr_uint16_batch_left_input_data
    del np_arr_uint16_batch_right_input_data

    # Sub-job #4
    fn_subjob4_to_save_results(
        bytearray_batch_output_data = bytearray_batch_output_data,
        str_output_file_name = str_output_file_name,)

    # Package job output results
    if len(args) == 3 and isinstance(args[2], list) :
        # Single process case (either single- or multithreaded).
        # Lock is unnecessary, since just 1 job-specific list element is updated.
        lst_tpl_jobs_outputs = args[2]
        lst_tpl_jobs_outputs[int_job_index] = (int_batch_output_data_checksum,)
    elif len(args) == 5 and \
         isinstance(args[3], multiprocessing.managers.NamespaceProxy) and \
         isinstance(args[4], multiprocessing.synchronize.Lock) :
        # Multiprocess case.
        # Ignore args[2] with lst_tpl_jobs_outputs in case of multiprocessing!
        multiproc_manager_namespace = args[3]
        multiproc_lock = args[4]
        # In case of multiprocessing, direct update of the mutable object in
        # the managed container does not work (!):
        # multiproc_manager_namespace.lst_tpl_jobs_outputs[int_job_index] = \
        #     (int_batch_output_data_checksum,)
        multiproc_lock.acquire()
        try:
            lst_tpl_jobs_outputs = multiproc_manager_namespace.lst_tpl_jobs_outputs
            lst_tpl_jobs_outputs[int_job_index] = (int_batch_output_data_checksum,)
            multiproc_manager_namespace.lst_tpl_jobs_outputs = lst_tpl_jobs_outputs
        finally:
            multiproc_lock.release()


def run_jobs_sequentially(lst_tpl_jobs_inputs) :

    int_jobs_count = len(lst_tpl_jobs_inputs)
    lst_tpl_jobs_outputs = [None] * int_jobs_count # allocate list of outputs per job

    print("Starting " + str(int_jobs_count) + " jobs SEQUENTIALLY...")
    print()
    #time_before_jobs_started = time.time()
    int_start_timestamp_nanoseconds = time.monotonic_ns()

    for int_job_index in range(int_jobs_count) :
        fn_job(
            int_job_index,
            lst_tpl_jobs_inputs[int_job_index],
            lst_tpl_jobs_outputs,)
        print("\nJob '" + str(int_job_index) + "' is finished at " +
              datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
              flush = True, end = "")

    #time_after_jobs_completed = time.time()
    #flt_time_interval_secs = time_after_jobs_completed - time_before_jobs_started
    int_end_timestamp_nanoseconds = time.monotonic_ns()
    int_script_runtime_nanoseconds = int_end_timestamp_nanoseconds - \
        int_start_timestamp_nanoseconds
    int_script_runtime_milliseconds = int(round(float(
        int_script_runtime_nanoseconds) / INT_NANOSECONDS_PER_MILLISECOND))

    print()
    print()
    print("REPORT: ")
    print(("1. The total runtime for all {:d} SEQUENTIAL jobs was {:,} milliseconds.".format(
              int_jobs_count, int_script_runtime_milliseconds)))
    print("2. All " + str(int_jobs_count) + " sequential jobs are completed.")
    print("3. Only main process was used.")
    print("4. Job queue was not used.")
    print()

    return lst_tpl_jobs_outputs


def start_daemon_worker_processes(
        multiproc_manager_namespace,
        int_worker_processes_count,
        int_max_jobs_queue_size,) :

    queue_parallel_jobs = multiprocessing.JoinableQueue(
        maxsize = int_max_jobs_queue_size) # FIFO
    multiproc_lock = multiprocessing.Lock()
    for int_process_index in range(int_worker_processes_count) :
        multiprocessing.Process(
            target = fn_worker_process,
            args = (
                int_process_index,
                queue_parallel_jobs,
                multiproc_manager_namespace,
                multiproc_lock),
            daemon = True).start()
    # There is no need to track these daemon processes manually:
    # they are managed through the multiprocessing.JoinableQueue.
    return (queue_parallel_jobs, multiproc_lock)


def run_jobs_parallelly(
        lst_tpl_jobs_inputs,
        int_worker_processes_count,
        int_max_jobs_queue_size,) :

    int_jobs_count = len(lst_tpl_jobs_inputs)
    lst_tpl_jobs_outputs = [(None,)] * int_jobs_count # allocate list of outputs per job

    multiproc_manager = multiprocessing.Manager()
    multiproc_manager_namespace = multiproc_manager.Namespace()
    multiproc_manager_namespace.lst_tpl_jobs_outputs = lst_tpl_jobs_outputs
    (queue_parallel_jobs, multiproc_lock) = start_daemon_worker_processes(
        multiproc_manager_namespace = multiproc_manager_namespace,
        int_worker_processes_count = int_worker_processes_count,
        int_max_jobs_queue_size = int_max_jobs_queue_size,)

    print("Starting " + str(int_jobs_count) + " jobs in PARALLEL ...")
    #time_before_jobs_started = time.time()
    int_start_timestamp_nanoseconds = time.monotonic_ns()

    for int_job_index in range(int_jobs_count) :
        str_job_name = str(int_job_index)
        dict_thread_params = \
        {
            "target" : fn_job,
            "name" : str_job_name,
            "args" : (
                int_job_index,
                lst_tpl_jobs_inputs[int_job_index],
                None, # lst_tpl_jobs_outputs,
                ),
        }
        # An attempt to "put" Thread instead causes and error:
        # "cannot pickle a lock object".
        queue_parallel_jobs.put(dict_thread_params,
            block = True, # will block if queue is full, unblocked by daemons
            timeout = None, # no timeout
            )
        print("\nMain Process added Job '{}' to the queue at ".format(str_job_name) +
              datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
              flush = True, end = "")

    print(flush = True)
    queue_parallel_jobs.join()
    lst_tpl_jobs_outputs = multiproc_manager_namespace.lst_tpl_jobs_outputs

    #time_after_jobs_completed = time.time()
    #flt_time_interval_secs = time_after_jobs_completed - time_before_jobs_started
    int_end_timestamp_nanoseconds = time.monotonic_ns()
    int_script_runtime_nanoseconds = int_end_timestamp_nanoseconds - \
        int_start_timestamp_nanoseconds
    int_script_runtime_milliseconds = int(round(float(
        int_script_runtime_nanoseconds) / INT_NANOSECONDS_PER_MILLISECOND))

    print()
    print("REPORT: ")
    print(("1. The total runtime for all {:d} PARALLEL jobs was {:,} milliseconds.".format(
              int_jobs_count, int_script_runtime_milliseconds)))
    print("2. All " + str(int_jobs_count) + " parallel jobs are completed.")
    print("3. Up to " + str(int_worker_processes_count) +
          " parallel daemon process(es) were/was used.")
    print("4. Job queue had a maximum size of " + str(int_max_jobs_queue_size) + ".")

    # Close all running daemon processes explicitly by passing dummy jobs.
    close_daemon_worker_processes(
        int_worker_processes_count = int_worker_processes_count,
        queue_parallel_jobs = queue_parallel_jobs,)

    return lst_tpl_jobs_outputs


def close_daemon_worker_processes(
        int_worker_processes_count, queue_parallel_jobs) :
    print("\nStarting to close " + str(int_worker_processes_count) +
          " daemon worker processes (optional)...")
    for int_process_index in range(int_worker_processes_count) :
        str_job_name = "Exit"
        dict_thread_params = {
            "target" : fn_job,
            "name" : str_job_name,
            "args" : (),
            }
        # An attempt to "put" Thread instead causes and error:
        # "cannot pickle a lock object".
        queue_parallel_jobs.put(dict_thread_params,
            block = True, # will block if queue is full, unblocked by daemons
            timeout = None # no timeout
            )
        print("\nMain Process added Job '{}' to the queue at ".format(str_job_name) +
            datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            flush = True, end="")
    print(flush = True)
    queue_parallel_jobs.join() # Wait till all daemons processes exit.

    print("\nAll daemon processes have been closed explicitly as of " +
          datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3] + "\n",
          flush = True,)


def package_jobs_inputs(
        str_left_input_file_name,
        str_right_input_file_name,
        str_trace_file_name_prefix,
        str_output_file_name,
        int_netto_input_row_length_bytes,
        int_netto_output_row_length_bytes,
        int_jobs_count,) :

    int_left_input_file_size_in_bytes = os.stat(str_left_input_file_name).st_size
    print('The size of the left input file "' + str_left_input_file_name + '" is ' +
          str(int_left_input_file_size_in_bytes) + ' bytes.', flush = True,)
    int_right_input_file_size_in_bytes = os.stat(str_right_input_file_name).st_size
    print('The size of the right input file "' + str_right_input_file_name + '" is ' +
          str(int_right_input_file_size_in_bytes) + ' bytes.', flush = True,)
    print(flush = True,)

    lst_tpl_jobs_inputs = []
    if int_left_input_file_size_in_bytes == int_right_input_file_size_in_bytes :
        int_job_index = 0
        lst_tpl_jobs_inputs = [None] * int_jobs_count # allocate list of inputs per job
        for batch_slice in sklearn.utils.gen_even_slices(
                n = int_left_input_file_size_in_bytes,
                n_packs = int_jobs_count,) :
            lst_tpl_jobs_inputs[int_job_index] = (
                str_left_input_file_name,
                str_right_input_file_name,
                str_trace_file_name_prefix,
                str_output_file_name,
                int_netto_input_row_length_bytes,
                int_netto_output_row_length_bytes,
                batch_slice.start,
                batch_slice.stop,)
            int_job_index += 1
    else :
        print("ERROR: " +
              "the size of the left file is different " +
              "from the one of the right one.")
        print(flush = True,)
    return lst_tpl_jobs_inputs


def obtain_inputs() :

    # file1.txt
    str_left_input_file_name = input("Enter the left input file name: ")
    # file2.txt
    str_right_input_file_name = input("Enter the right input file name: ")
    # tracefile
    str_trace_file_name_prefix = input("Enter the trace file name prefix: ")
    # newfile1.txt, newfile2.txt, newfile5.txt, newfile10.txt, or newfile20.txt.
    str_output_file_name = input("Enter the output file name: ")

    int_jobs_count = -1
    while int_jobs_count < 1 :
        # 1, 2, 5, 10, or 20.
        str_jobs_count = input("Enter the total count of jobs ( >= 1): ")
        if str_jobs_count.isdigit() :
            int_jobs_count = int(str_jobs_count)
    int_worker_processes_count = -1
    int_max_jobs_queue_size = -1
    if int_jobs_count > 1 :
        while int_worker_processes_count < 1 :
            # try different values as a ways to increase efficiency and
            # reduce processing time.
            str_worker_processes_count = input(
                "Enter the total count of worker processes ( >= 1): ")
            if str_worker_processes_count.isdigit() :
                int_worker_processes_count = int(str_worker_processes_count)
        while int_max_jobs_queue_size < 1 :
            # try different values as a ways to increase efficiency and
            # reduce processing time.
            str_max_jobs_queue_size = input(
                "Enter the maximum job queue size ( >= 1): ")
            if str_max_jobs_queue_size.isdigit() :
                int_max_jobs_queue_size = int(str_max_jobs_queue_size)
    else :
        int_worker_processes_count = 1
        int_max_jobs_queue_size = 0
    print(flush = True,)

    return (
        str_left_input_file_name, str_right_input_file_name,
        str_trace_file_name_prefix, str_output_file_name,
        int_jobs_count, int_worker_processes_count, int_max_jobs_queue_size,)


def run_jobs(
        str_left_input_file_name,
        str_right_input_file_name,
        str_trace_file_name_prefix,
        str_output_file_name,
        int_integer_maximum_input_value,
        int_integer_maximum_output_value,
        int_jobs_count,
        int_worker_processes_count = 1,
        int_max_jobs_queue_size = 1) :

    int_netto_input_row_length_bytes = len(str(int_integer_maximum_input_value))
    int_netto_output_row_length_bytes = len(str(int_integer_maximum_output_value))

    lst_tpl_jobs_inputs = package_jobs_inputs(
        str_left_input_file_name = str_left_input_file_name,
        str_right_input_file_name = str_right_input_file_name,
        str_trace_file_name_prefix = str_trace_file_name_prefix,
        str_output_file_name = str_output_file_name,
        int_netto_input_row_length_bytes = int_netto_input_row_length_bytes,
        int_netto_output_row_length_bytes = int_netto_output_row_length_bytes,
        int_jobs_count = int_jobs_count,)

    try :
        os.unlink(str_output_file_name)
    except :
        pass

    if int_worker_processes_count == 1 :
        lst_tpl_jobs_outputs = run_jobs_sequentially(
            lst_tpl_jobs_inputs = lst_tpl_jobs_inputs,)
    elif int_worker_processes_count > 1 :
        lst_tpl_jobs_outputs = run_jobs_parallelly(
            lst_tpl_jobs_inputs = lst_tpl_jobs_inputs,
            int_worker_processes_count = int_worker_processes_count,
            int_max_jobs_queue_size = int_max_jobs_queue_size,)

    print("All jobs/batches checksums:\n", flush = True,)
    int_grand_total_checksum = 0
    for int_job_index in range(len(lst_tpl_jobs_outputs)) :
        int_checksum = lst_tpl_jobs_outputs[int_job_index][0]
        int_grand_total_checksum += int_checksum
        print("Job '" + str(int_job_index) + "' checksum: " + str(int_checksum))
    print("Grand Total Checksum for All Jobs: " + str(int_grand_total_checksum))


def main() :

    if True :
        (str_left_input_file_name, str_right_input_file_name,
         str_trace_file_name_prefix, str_output_file_name,
         int_jobs_count, int_worker_processes_count, int_max_jobs_queue_size) = \
            obtain_inputs()

    if False :
        (str_left_input_file_name, str_right_input_file_name,
         str_trace_file_name_prefix, str_output_file_name,
         int_jobs_count, int_worker_processes_count, int_max_jobs_queue_size) = \
            ("file1.txt", "file2.txt",
             "tracefile", "newfile_1_1_1.txt", 1, 1, 1,)

    if False :
        (str_left_input_file_name, str_right_input_file_name,
         str_trace_file_name_prefix, str_output_file_name,
         int_jobs_count, int_worker_processes_count, int_max_jobs_queue_size) = \
            ("file1.txt", "file2.txt",
             "tracefile", "newfile_2_2_2.txt", 2, 2, 2,)

    if False :
        (str_left_input_file_name, str_right_input_file_name,
         str_trace_file_name_prefix, str_output_file_name,
         int_jobs_count, int_worker_processes_count, int_max_jobs_queue_size) = \
            ("file1.txt", "file2.txt",
             "tracefile", "newfile_10_10_10.txt", 10, 10, 10,)

    if False :
        (str_left_input_file_name, str_right_input_file_name,
         str_trace_file_name_prefix, str_output_file_name,
         int_jobs_count, int_worker_processes_count, int_max_jobs_queue_size) = \
            ("hugefile1.txt", "hugefile2.txt",
             "tracefile", "totalfile.txt", 2, 2, 2,)

    if False :
        (str_left_input_file_name, str_right_input_file_name,
         str_trace_file_name_prefix, str_output_file_name,
         int_jobs_count, int_worker_processes_count, int_max_jobs_queue_size) = \
            ("hugefile1.txt", "hugefile2.txt",
             "tracefile", "totalfile.txt", 10, 10, 10,)

    run_jobs(
        str_left_input_file_name = str_left_input_file_name,
        str_right_input_file_name = str_right_input_file_name,
        str_trace_file_name_prefix = str_trace_file_name_prefix,
        str_output_file_name = str_output_file_name,
        int_integer_maximum_input_value = INT_INTEGER_MAXIMUM_INPUT_VALUE,
        int_integer_maximum_output_value = INT_INTEGER_MAXIMUM_OUTPUT_VALUE,
        int_jobs_count = int_jobs_count,
        int_worker_processes_count = int_worker_processes_count,
        int_max_jobs_queue_size = int_max_jobs_queue_size,)


if __name__ == "__main__" :
    main()

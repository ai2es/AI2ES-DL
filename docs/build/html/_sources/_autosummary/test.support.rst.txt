test.support
============

.. automodule:: test.support

   
   
   

   
   
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   
      anticipate_failure
      args_from_interpreter_flags
      bigaddrspacetest
      bigmemtest
      calcobjsize
      calcvobjsize
      can_symlink
      can_xattr
      captured_output
      captured_stderr
      captured_stdin
      captured_stdout
      change_cwd
      check__all__
      check_free_after_iterating
      check_impl_detail
      check_no_resource_warning
      check_no_warnings
      check_sanitizer
      check_sizeof
      check_syntax_error
      check_syntax_warning
      check_warnings
      clear_ignored_deprecations
      collision_stats
      cpython_only
      create_empty_file
      darwin_malloc_err_warning
      detect_api_mismatch
      disable_faulthandler
      disable_gc
      fd_count
      findfile
      forget
      fs_is_case_insensitive
      gc_collect
      get_attribute
      get_original_stdout
      ignore_deprecations_from
      ignore_warnings
      impl_detail
      import_fresh_module
      import_module
      infinite_recursion
      is_resource_enabled
      join_thread
      load_package_tests
      make_bad_fd
      make_legacy_pyc
      match_test
      maybe_get_event_loop_policy
      missing_compiler_executable
      modules_cleanup
      modules_setup
      no_tracing
      open_dir_fd
      open_urlresource
      optim_args_from_interpreter_flags
      patch
      print_warning
      python_is_optimized
      reap_children
      reap_threads
      record_original_stdout
      refcount_test
      requires
      requires_bz2
      requires_freebsd_version
      requires_gzip
      requires_linux_version
      requires_lzma
      requires_mac_ver
      requires_resource
      requires_zlib
      rmdir
      rmtree
      run_doctest
      run_in_subinterp
      run_unittest
      run_with_locale
      run_with_tz
      save_restore_warnings_filters
      set_match_tests
      set_memlimit
      setswitchinterval
      skip_if_broken_multiprocessing_synchronize
      skip_if_buggy_ucrt_strfptime
      skip_if_new_parser
      skip_if_pgo_task
      skip_if_sanitizer
      skip_unless_symlink
      skip_unless_xattr
      sortdict
      start_threads
      suppress_msvcrt_asserts
      swap_attr
      swap_item
      system_must_validate_cert
      temp_cwd
      temp_dir
      temp_umask
      threading_cleanup
      threading_setup
      unlink
      unload
      use_old_parser
      wait_process
      wait_threads_exit
      with_pymalloc
   
   

   
   
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :template: custom-class-template.rst
   
      BasicTestRunner
      CleanImport
      DirsOnSysPath
      EnvironmentVarGuard
      FakePath
      Matcher
      PythonSymlink
      SaveSignals
      SuppressCrashReport
      TransientResource
      WarningsRecorder
      catch_threading_exception
      catch_unraisable_exception
   
   

   
   
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
   
      Error
      ResourceDenied
      TestDidNotRun
      TestFailed
      TestFailedWithDetails
   
   



.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:

   test.support.bytecode_helper
   test.support.hashlib_helper
   test.support.logging_helper
   test.support.script_helper
   test.support.socket_helper
   test.support.testresult
   test.support.warnings_helper


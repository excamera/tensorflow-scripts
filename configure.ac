#                                               -*- Autoconf -*-
# Process this file with ./autogen.sh to produce a configure script

AC_PREREQ([2.68])
AC_INIT([tensorflow-scripts], [0.01], [kvasukib@eng.ucsd.edu])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_SRCDIR([src/])

# Checks for programs.

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

# Check for Python
AM_PATH_PYTHON(2.7)
AM_CHECK_PYTHON_HEADERS(,[AC_MSG_ERROR(could not find Python headers)])

AC_CONFIG_FILES([Makefile
		 src/Makefile])

AC_OUTPUT

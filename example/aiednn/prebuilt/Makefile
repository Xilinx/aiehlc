###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# An app on the prebuilt aiednn library: build the library once (python -m aiegraph dnn-lib), then
#   make DIST=<dist>    (default: worklocal/aiednn/lib_vek280)
DIST ?= $(abspath ../../../worklocal/aiednn/lib_vek280)
include $(DIST)/link.mk

app.elf: app.o
	$(AIEDNN_CXX) -o $@ app.o $(AIEDNN_LDFLAGS) $(AIEDNN_LIBS)
app.o: app.cc
	$(AIEDNN_CXX) $(AIEDNN_CXXFLAGS) -c -o $@ $<
clean:
	rm -f app.o app.elf

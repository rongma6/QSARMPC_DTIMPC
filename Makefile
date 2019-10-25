
BASE_FLAG ?= --compilation_mode=opt --incompatible_depset_is_not_iterable=False --copt=-O3 --cxxopt='-std=c++17'

TARGET = //:client //controller:main-dsc //controller:main-tc //:es //:debug //:run

ifneq ($(local), y)
BASE_FLAG += --config=bbb-privpy-builder
endif

ifeq ($(auth), y)
    BASE_FLAG += --cxxopt=-DAUTH
endif

ifeq ($(brpc), y)
    BASE_FLAG += --cxxopt=-DBRPC
    BASE_FLAG += --define rpc=brpc
endif

ifeq ($(openmp), n)
    BASE_FLAG += --define parallel=noopenmp
endif

ifeq ($(sim_sdk), y)
    BASE_FLAG += --define SIMULATE_SDK=1
endif

#########enable use crypto with hardware################
ifeq ($(sm), y)
    BASE_FLAG += --cxxopt=-DSM
endif
########################################################

kill:
	killall -9 run.py client.py main run.par es.par debug.par client.par es_wo_lib.par main-sgx main-cache main-tc main-dsc debug_server.py run.py es.py

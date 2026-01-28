SCRIPT_DIR=$(dirname $(readlink -f "${BASH_SOURCE[0]}"))
export TL_ROOT=/home/developer/workspace/git/github/tile-ai/tilelang-ascend

rm -rf ${SCRIPT_DIR}/test_elementwise_add.so
bisheng \
    --npu-arch=dav-2201 \
    -O2 -std=c++17 -xasc \
    -I${ASCEND_HOME_PATH}/include \
    -I${ASCEND_HOME_PATH}/include/experiment/msprof \
    -I${ASCEND_HOME_PATH}/include/experiment/runtime \
    -I${ASCEND_HOME_PATH}/pkg_inc \
    -I${ASCEND_HOME_PATH}/pkg_inc/runtime \
    -I${ASCEND_HOME_PATH}/pkg_inc/profiling \
    -I${TL_ROOT}/src \
    -I${TL_ROOT}/3rdparty/catlass/include \
    -I${TL_ROOT}/3rdparty/shmem/include \
    -I${TL_ROOT}/3rdparty/shmem/src/device \
    -L${ASCEND_HOME_PATH}/lib64 \
    -Wno-macro-redefined -Wno-ignored-attributes -Wno-non-c-typedef-for-linkage \
    -lruntime -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl \
    -fPIC --shared ${SCRIPT_DIR}/test_elementwise_add.cpp -o ${SCRIPT_DIR}/test_elementwise_add.so

echo "Compile Success."
python ${SCRIPT_DIR}/test_elementwise_add.py
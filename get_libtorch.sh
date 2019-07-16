LIBTORCH_LOCATION=$1
echo "Copying libtorch from ${LIBTORCH_LOCATION}"

mkdir libtorch
rsync -arv ${LIBTORCH_LOCATION}/lib libtorch
rsync -arv ${LIBTORCH_LOCATION}/bin libtorch
rsync -arv ${LIBTORCH_LOCATION}/share libtorch
rsync -arv ${LIBTORCH_LOCATION}/include libtorch


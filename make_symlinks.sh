#!/bin/sh

[ ! -d target ] && mkdir target || rm -rf target/*
[ ! -d diff ] && mkdir diff || rm -rf diff/*
[ ! -d samples_PN ] && mkdir samples_PN || rm -rf samples_PN/*
[ ! -d samples_AN ] && mkdir samples_AN || rm -rf samples_AN/*

for img in *_target.png ; 
do
	[ ! -e "target/$img" ] && ln -s -r $img target/
done

for img in *_diff.png ; 
do
	[ ! -e "diff/$img" ] && ln -s -r $img diff/ 
done

for img in *_samples_PN.png;
do
	[ ! -e "samples_PN/$img" ] && ln -s -r $img samples_PN/ 
done

for img in *_samples_AN.png ;
do
	[ ! -e "samples_AN/$img" ] && ln -s -r $img samples_AN/
done


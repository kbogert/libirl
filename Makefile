

test/libirl-test: liblibirl.a test/source/*.d
	dub build --build=unittest --root=test

liblibirl.a: source/*.d
	dub build

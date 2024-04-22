Changes that can be automated when updating from PaRSEC API v3.x to PaRSEC API v4.x
===================================================================================

With the 4.0 release, PaRSEC has undergone a major renaming in an effort to properly namespace symbols provided by PaRSEC.
This will likely affect existing codes that will have to adapt to using the new symbols.
To help with this effort, the sed script `contrib/renaming/rename.sed` is provided that can be used to apply the changes to existing source code files.
To apply the changes, users can execute the following command:

```sh
find . -name "*.c" -or -name "*.h" -or -name "*.cc" -or -name "*.jdf" -or -name "*.h.in" | xargs sed -i -f $PARSEC_SOURCE_DIR/contrib/renaming/rename.sed
```

Depending on your source code, your may have to adapt the above line to apply to all relevant files.

The full list of replacements can be found below:

```sh
# data types
enum matrix_type -> parsec_matrix_type_t
matrix_RealDouble -> PARSEC_MATRIX_DOUBLE
matrix_RealFloat -> PARSEC_MATRIX_FLOAT
matrix_Integer -> PARSEC_MATRIX_INTEGER
matrix_Byte -> PARSEC_MATRIX_BYTE
matrix_ComplexFloat -> PARSEC_MATRIX_COMPLEX_FLOAT
matrix_ComplexDouble -> PARSEC_MATRIX_COMPLEX_DOUBLE

# storage
enum matrix_storage -> parsec_matrix_storage_t
matrix_Lapack -> PARSEC_MATRIX_LAPACK
matrix_Tile -> PARSEC_MATRIX_TILE

# matrix shapes
enum matrix_uplo -> parsec_matrix_uplo_t
matrix_UpperLower -> PARSEC_MATRIX_FULL
matrix_Upper -> PARSEC_MATRIX_UPPER
matrix_Lower -> PARSEC_MATRIX_LOWER

# vector shapes
enum vector_distrib -> parsec_vector_two_dim_cyclic_distrib_t
matrix_VectorRow -> PARSEC_VECTOR_DISTRIB_ROW
matrix_VectorCol -> PARSEC_VECTOR_DISTRIB_COL
matrix_VectorDiag -> PARSEC_VECTOR_DISTRIB_DIAG

# matrix implementations
parsec_tiled_matrix_dc* -> parsec_tiled_matrix*
sym_two_dim_block_cyclic* -> parsec_matrix_sym_block_cyclic*
two_dim_block_cyclic* -> parsec_matrix_block_cyclic*
two_dim_tabular_* -> parsec_matrix_tabular_*
two_dim_td_table_clone_table_structure -> parsec_matrix_tabular_clone_table_structure

# symbols in matrix.h
tiled_matrix_submatrix -> parsec_tiled_matrix_submatrix
parsec_matrix_create_data -> parsec_tiled_matrix_create_data
parsec_matrix_add2arena -> parsec_matrix_adt_construct
parsec_add2arena -> parsec_matrix_adt_construct
parsec_matrix_del2arena -> parsec_matrix_arena_datatype_destruct_free_type
parsec_del2arena -> parsec_matrix_arena_datatype_destruct_free_type
parsec_matrix_data_* -> parsec_tiled_matrix_data_*

# vector
vector_two_dim_cyclic_* -> parsec_vector_two_dim_cyclic*

# 2D grid
grid_2Dcyclic_* -> parsec_grid_2Dcyclic_*

# matrix ops
tiled_matrix_unary_op_t -> parsec_tiled_matrix_unary_op_t
tiled_matrix_binary_op_t -> parsec_tiled_matrix_binary_op_t

# arena_datatype contructors/destructors
parsec_dtd_destroy_arena_datatype -> parsec_dtd_free_arena_datatype
```

Changes that are not automated when updating between PaRSEC API v3.x and v4.x
=============================================================================

DTD initialization of arena datatypes has changed to avoid memory leakage.
--------------------------------------------------------------------------

We show below some typical conversions. More can be found by examining the commit that modified this line.

This snippet extracted from `contrib/build_with_parsec/dtd_text_allreduce.c` shows both the v3.x and v4.x canonical snippet:
```c
    // v3.x to v4.0 release: DTD arena_datatype allocation and initialization
#if PARSEC_VERSION_MAJOR < 4
    parsec_add2arena_rect(&parsec_dtd_arenas_datatypes[TILE_FULL],
        parsec_datatype_int32_t, nb, 1, nb);
#else
    parsec_arena_datatype_t *adt = PARSEC_OBJ_NEW(parsec_arena_datatype);
    parsec_matrix_adt_construct_rect(adt,
        parsec_datatype_int32_t, nb, 1, nb);
    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);
#endif
```

If you had updated to pre-release v4.x, you may have to do the following change:
```diff
     // v4.x pre-release to v4.0 release: DTD arena_datatype allocation and initialization
-    parsec_arena_datatype_t *adt;
-    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
-    parsec_add2arena_rect( adt,
-       parsec_datatype_int32_t, nb, 1, nb);
+    parsec_arena_datatype_t *adt = PARSEC_OBJ_NEW(parsec_arena_datatype);
+    parsec_matrix_adt_construct_rect(adt,
+       parsec_datatype_int32_t, nb, 1, nb);
+    parsec_dtd_attach_arena_datatype(parsec, adt, &TILE_FULL);
```

The cleanup of DTD allocated arena datatypes can be significantly simplified:
```diff
     // v3.x to v4.0: DTD arena_datatype cleanup
-    adt = parsec_dtd_get_arena_datatype(parsec, TILE_FULL);
-    assert(NULL != adt);
-    parsec_type_free(&adt->opaque_dtt);
-    PARSEC_OBJ_RELEASE(adt->arena);
-    parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
+    adt = parsec_dtd_detach_arena_datatype(parsec, TILE_FULL);
+    assert(NULL != adt);
+    parsec_type_free(&adt->opaque_dtt);
+    PARSEC_OBJ_RELEASE(adt);
```

This snippet repeated itself often enough that we have a shorthand for detaching the adt,
clearing the type, and releasing the adt all in one go:
```diff
     // equivalent to the new code above
+    parsec_dtd_free_arena_datatype(parsec, DATA);
```
PTG use of del2arena
--------------------

In some PTG wrappers, v3.x canonical approach would have a `parsec_del2arena(adt)`
call followed by `PARSEC_OBJ_RELEASE(adt->arena)`.

in v4.x the `parsec_arena_datatype_t` destructor recursively destructs the internal members of the `adt`. Hence the `PARSEC_OBJ_RELEASE(adt->arena)` is not necessary anymore. (legacy code that still call the explicit release remain correct, the adt destructor will care not to cause a double free).

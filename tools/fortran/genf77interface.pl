#!/usr/bin/perl

use strict;

my $input = "inputf77.c";
my $output = "dplasma_zf77.c";

sub printHeader {

    print OUTFILE "/**\n";
    print OUTFILE " *\n";
    print OUTFILE " * \@file dplasma_zf77.c\n";
    print OUTFILE " *\n";
    print OUTFILE " *  DPLASMA Fortran 77 interface for computational routines\n";
    print OUTFILE " *  DPLASMA is a software package provided by Univ. of Tennessee,\n";
    print OUTFILE " *  Univ. of California Berkeley and Univ. of Colorado Denver\n";
    print OUTFILE " *\n";
    print OUTFILE " * \@version 1.0.0\n";
    print OUTFILE " * \@author Mathieu Faverge\n";
    print OUTFILE " * \@date 2011-12-05\n";
    print OUTFILE " * \@precisions normal z -> c d s\n";
    print OUTFILE " *\n";
    print OUTFILE " **/\n";
    print OUTFILE "#include \"parsec.h\"\n";
    print OUTFILE "#include <plasma.h>\n";
    print OUTFILE "#include \"dplasma.h\"\n";
    print OUTFILE "#include \"dplasmaf77.h\"\n";
    print OUTFILE "#include \"data_dist/matrix/matrix.h\"\n";
    print OUTFILE "\n\n";

}

sub PrintFile {
    
    my ( $define, $functions ) = @_ ;

    printHeader();
    print OUTFILE $define;
    print OUTFILE $functions;

}

sub ParseCore {
    
    my ($file) = @_ ;

    my $define = "";
    my $functions = "";

    open(M, $file);
    while( ) 
    {
        my $line = <M>;
        chomp $line;
        if ( ! ($line =~ /dplasma_/ ) ) {
            $define .= $line."\n";
            $functions .= $line."\n";
            last if eof(M);
            next;
        }
        
        my $fullline = $line;
        while ( ! ($line =~ /\);/ ) ) {
            $line = <M>;
            chomp $line;
            $fullline .= $line;
        }

        # Remove the const
        $fullline =~ s/const //g;
        $fullline =~ s/\* / */g;
        $fullline =~ s/[\t ]+/ /g;
        $fullline =~ s/\s*\)/)/g;

        print "Input:\n$fullline\n";

        # Remove the "parsec_context_t *parsec"
        $fullline =~ s/parsec_context_t \*parsec, //g;

        # Get the type return by the function
        my $rettype = $fullline;
        $rettype =~ s/(.*) dplasma_.*/\1/;

        my $funcname = $fullline;
        $funcname =~ s/.*dplasma_z(\w*)\s*\(.*/\1/;
        
        # duplicate the list of parameter
        my $param = $fullline;
        $param =~ s/.*(\(.*\));/\1/;
        $param =~ s/[^,(]*\s\**([a-zA-Z0-9_]*[,)])/\1 /g; #Remove types
        $param =~ s/([\s(])(\w*[,)])/\1*\2/g; #Remove types

        my $call = "dplasma_z${funcname}${param};";
        $call =~ s/\(/( parsecf77_context, /;

        $fullline =~ s/(\**)([a-zA-Z0-9_]*[,)])/\1*\2/g;
        $fullline =~ s/;//;
        $fullline =~ s/^${rettype}/void/;
        $fullline =~ s/dplasma_/dplasmaf77_/;

        if ( !( $rettype =~ /void/ ) ) {
            $fullline =~ s/\)/, ${rettype} *ret )/;
            $call = "*ret = $call\n";
        }
                  
        print "Return:\n$rettype\n";
        print "Fullline:\n$fullline\n";
        print "Param:\n$param\n";

        # Let's print the result
        $define .= "#define dplasmaf77_z${funcname}    DPLASMA_ZF77NAME( ${funcname}, ".uc($funcname)." )\n";
        $functions .= $fullline."\n"
            ."{\n"
            ."    extern parsec_context_t *parsecf77_context;\n"
            ."    ".$call."\n"
            ."}\n\n";

        last if eof(M);
    }

    printHeader();
    print OUTFILE $define;
    print OUTFILE "\n\n";
    print OUTFILE $functions;

}

open(OUTFILE, ">$output");
ParseCore( $input );
close(OUTFILE);

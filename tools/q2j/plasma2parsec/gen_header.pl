#!/usr/bin/perl

use strict;

my @files  = `ls -1 core_z*.c`;
my $output = "core.h";

sub ParseCore {

    my ($file) = @_ ;

    open(M, $file);
    while( )
    {
        my $line = <M>;
        chomp $line;
        if ( $line =~ /QUARK_CORE/ ) {
            my $pragma = "#pragma ";
            while ( ! ($line =~ /{/ ) ) {
                $line .= <M>;
                chomp $line;
            }
            $line =~ s/[^,(]*\s\**([a-zA-Z0-9_]*[,)])/\1 /g; #Remove type
            $line =~ s/[ \t]+/ /g;                           #Remove spaces and tab
            $line =~ s/{[ \t]*/{\\\n/g;                      #Replace { by \\n
            $line =~ s/\w* *(\w*\()/#define \1/;             #Replace void by define
            print OUTPUT $line ;

            $line =~ s/#define QUARK_(.*)\(.*/\1/;
            chomp $line;
            $pragma .= $line;

            $line = <M>;
            chomp $line;
            while ( ! ($line =~ /0\);/ ) ) {
                if ( $line =~ /.*,[ ]*(.*),[ ]*[INOPUT]*,/ ) {
                    $pragma .= " ".$1;
                }
                if ($line =~ /QUARK_Insert_Task/ ) {
                    $line =~ s/(\(quark)/(\1)/g;
                    $line =~ s/(task_flags)/(\1)/g;
                } else {
                    $line =~ s/(.*,[ \t]+&?)(\w*)(,.*,)/\1(\2)\3/g;
                }
                $line .= "\\\n";
                print OUTPUT $line;


                $line = <M>;
                chomp $line;
            }
            # Last line with '0);'
            $line .= "}\n";
            print OUTPUT $line;
            print OUTPUT $pragma."\n\n";
        }
        last if eof(M);
    }
    close(M);
}

#################################################################################
#
#                   Main
#
#################################################################################

open(OUTPUT, ">$output");
foreach my $file (@files)
{
    chomp $file;
    #print "$file\n";
    ParseCore( $file );
}
close(OUTPUT);

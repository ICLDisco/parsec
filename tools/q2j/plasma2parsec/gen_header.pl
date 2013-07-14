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
        my $output = "";

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
            $output .= $line ;

            $line =~ s/#define QUARK_(.*)\(.*/\1/;
            chomp $line;

            my $fname = $line;
            $fname =~ s/CORE_//;
            $pragma .= $fname;

            $line = <M>;
            chomp $line;
            while ( ! ($line =~ /0\);/ ) ) {
                if ($line =~ /QUARK_Task_Init/ ) {
                    goto end;
                }
                if ( ($line =~ /DAG_CORE_/) || ($line =~ /DAG_SET/) ) {
                    $line = <M>;
                    chomp $line;
                    next;
                }

                my $line_params = $line;
                $line_params =~ s/[ ]*\|[ ]*LOCALITY[ ]*//;
                $line_params =~ s/[ ]*\|[ ]*GATHERV[ ]*//;
                $line_params =~ s/[ ]*\|[ ]*QUARK_REGION_.*[ ]*//;
                #print $line_params."\n";
                if ( $line_params =~ /.*,[ ]*(.*),[ ]*[INOPUT]*,/ ) {
                    $pragma .= " ".$1;
                }
                if ($line =~ /QUARK_Insert_Task/ ) {
                    $line =~ s/(\(quark)/(\1)/g;
                    $line =~ s/(task_flags)/(\1)/g;
                } else {
                    $line =~ s/(.*,[ \t]+&?)(\w*)(,.*,)/\1(\2)\3/g;
                }
                $line .= "\\\n";
                $output .= $line;

                $line = <M>;
                chomp $line;
            }
            # Last line with '0);'
            $line .= "}\n";
            $output .= $line;
            $output .= $pragma."\n\n";
        }
        last if eof(M);
        print OUTPUT $output;
    }
end:
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
    print "$file\n";
    ParseCore( $file );
}
close(OUTPUT);

#!/usr/bin/perl

use strict;
my $verbose=0;
my %linesToNames;
my $currKernel;
my $lineno=0;

my $num=1+$#ARGV;
if( $num <= 1 ) {
   print "Usage: perl prettyPrint.pl PetitFile.t DepsFile.dep\n";
   exit;
}

my @stack;
my @paramSpaceStack;

print "TASK SECTION START\n";
open PETITFILE, "<", $ARGV[0];
while(my $line=<PETITFILE>){
    $lineno++;
    if( $line =~ /for *(\w*) *= *([^ ]+) *to *([^ ]+) *do/ ){
        push (@stack, $1);
        my $bounds = $1."=".$2."..".$3;
        push (@paramSpaceStack, $bounds);
    }
    if( $line =~ /endfor/ ){
        pop @stack;
        pop @paramSpaceStack;
    }

    # If we found the commend with the kernel name, parse it and keep
    # it for future use.
    if( $line =~ /\!\! *(\w*) *\(.*\)/ ){
        my $string = $1."(";
        my $i=0;
        foreach my $var (@stack){
            if( $i>0 ){
                $string .= ",";
            }
            $string .= $var;
            $i++;
        }
        $string .= ")";
        $linesToNames{$lineno}  = $string;

        # also print the kernel name and the parameter space on the
        # top of the output for the next stage to use.
        my $full_string = $string." {";
        my $i=0;
        foreach my $var (@paramSpaceStack){
            if( $i>0 ){
                $full_string .= ",";
            }
            $full_string .= $var;
            $i++;
        }
        $full_string .= "}";
        print "$full_string\n";
    }
}
close PETITFILE;
print "TASK SECTION END\n";

open DEPSFILE, "<", $ARGV[1];
while(<DEPSFILE>){
    my $ln=$_;
    my @words = split(/\s+/);
    if( $words[0] =~ /output|flow/ ){
        $verbose = 1;
        $words[1] =~ s/\://;
        my $num = int($words[1]);
        my $tmp = numToName($num);
        if( $currKernel ne $tmp ){
            $currKernel = $tmp;
            print "\n";
            print "########################################################################\n";
            print "### SOURCE: ",$currKernel,"\n";
        }
        print "\n";

        $words[4] =~ s/\://;
        my $num = int($words[4]);
        print "===> ", numToName($num), "\n"
    }
    if( $words[0] =~ /anti/ ){
        $verbose = 0;
    }
    if( $verbose ){
        print $ln;
    }
}

sub numToName{
    my $num = shift;
    my $key;
    my ($prevK, $prevV);

    foreach $key (sort {$a <=> $b} keys %linesToNames){
        if( int($key) <= $num ){
            $prevK = $key;
            $prevV = $linesToNames{$key};
        }else{
            last;
        }
    }
    return $prevV;
}

# flow    41: A(m,n)          -->  44: Exit                            [ M]
# {[k,n,m] -> [m,n] : 0 <= k < n,m < BB}
#
# output  41: A(m,n)          -->  10: A(k,k)          (+)             [ M]
# {[k,n,n] -> [n] : 0 <= k < n < BB}
# may dd: {[In_1]: 1 <= In_1}


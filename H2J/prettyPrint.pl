#!/usr/bin/perl

use strict;
my $verbose=0;
my %linesToNames;
my $currKernel;

# LU.t
$linesToNames{0}  = 'ENTRY';
$linesToNames{7}  = 'IN';
$linesToNames{17} = 'DGETRF';
$linesToNames{24} = 'DTSTRF';
$linesToNames{35} = 'DGESSM';
$linesToNames{43} = 'DSSSSM';
$linesToNames{55} = 'OUT';
$linesToNames{61} = 'EXIT';

# min.t
#$linesToNames{5}  = 'FRST';
#$linesToNames{9}  = 'SCND';

while(<>){
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

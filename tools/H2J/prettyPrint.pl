#!/usr/bin/perl

use strict;
use Switch;
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

# First parse the petit file and extract information about
# the tasks that is not preserved in petit's output

# void task_DSSSSM(double *C1,  double *C2, double *dL, double *L, int *IPIV)
# void task_DTSTRF(double *U, double *L, double *dL, int *IPIV)

# !       task_DTSTRF( A(k, k):U, A(m, k):LU, L(m, k), IPIV(m, k));
# !               INOUT       INOUT       INOUT       OUT      OUT
# !!      DTSTRF( A(k, k, 0), A(m, k, 0), A(m, k, 1), L(m, k), IPIV(m, k));

#convert it to
# !!      DTSTRF( U:A(k, k):U, L:A(m, k):LU, dL:L(m, k), IPIV:IPIV(m, k));

# TASK SECTION START
# IN(ii,jj) {ii=0..BB-1,jj=0..BB-1} 
# DGETRF(k) {k=0..BB-1} B:A(k, k, 0)|C:A(k, k, 1)|D:IPIV(k, k)
# DGESSM(k,n) {k=0..BB-1,n=k+1..BB-1} B:IPIV(k, k)|C:A(k, k, 1)|D:A(k, n, 0)|E:A(k, n, 1)
# DTSTRF(k,m) {k=0..BB-1,m=k+1..BB-1} B:A(k, k, 0)|C:A(m, k, 0)|D:A(m, k, 1)|E:L(m, k)|F:IPIV(m, k)
# DSSSSM(k,m,n) {k=0..BB-1,m=k+1..BB-1,n=k+1..BB-1} B:A(k, n, 0)|C:A(k, n, 1)|D:A(m, n, 0)|E:A(m, n, 1)|F:L(m, k)|G:A(
# m, k, 0)|H:A(m, k, 1)|I:IPIV(m, k)
# OUT(ii,jj) {ii=0..BB-1,jj=0..BB-1} 
# TASK SECTION END


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
    if( $line =~ /\!\! *(\w*) *\((.*)\)/ ){
        my $string = $1."(";
        my $args = $2;
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
        $full_string .= "} ";

        my $i=0;
        # parse the arguments and generate symbolic names for the arrays
        # An array will be the alias "\w*" followed by ":" followed by the
        # tile "\w*" followed by a l-paren "\(" followed by anything except
        # a r-paren "[^)]*" followed by a # r-paren "\)" optionally followed
        # by a ":L", or ":U", or ":LU" for lower, upper or both triangles
        while($args =~ /(\w*):(\w*\([^)]*\))(:(\w*))?/g){
            if( $i > 0 ){
                $full_string .= "|";
            }
            my $alias = $1;
            my $tile = $2;
            my $triangle = $4;
            if( !length $triangle ){
                $triangle = "LU";
            }

            switch( $triangle ){
                case "U"  { my $rPar = index($tile,")");
                            substr($tile,$rPar,1,", 0)");
                            $full_string .= $alias.":".$tile;
                          }
                case "L"  { my $rPar = index($tile,")"); 
                            substr($tile,$rPar,1,", 1)");
                            $full_string .= $alias.":".$tile;
                          }
                case "LU" { my $rPar = index($tile,")");
                            my $tmp = substr($tile,0,$rPar);
                            $full_string .= $alias."_u:".$tmp.", 0)|";
                            $full_string .= $alias."_l:".$tmp.", 1)";
                         }
            }
            $i++;
        }
        print "$full_string\n";
    }
}
close PETITFILE;
print "TASK SECTION END\n";

# The parse petit's output and organise the dependencies a little bit

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

# Helper function to convert line number (in petit file) to kernel name.
# Here is where the information we parsed in the first loop comes handy.
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


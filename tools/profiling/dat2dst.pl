#!/bin/perl -w

open(FIC, "sort -n -k 5 $ARGV[0]|") or die "Unable to open $ARGV[0]: $!\n";
my $dataleft = {};
my $id=0;
while( <FIC> ) {
  my $line = $_;
  next if( $line =~ /^#/ );
  my @v = split( "\t", $line);
  ${dataleft}->{"$v[2]$v[3]"} = $id++;
}
close(FIC);

my $distance = 0;

open(LOC, ">$ARGV[0]-$ARGV[1].total") or die "Unable to create $ARGV[0]-$ARGV[1].total";

open(FIC, "sort -n -k 5 $ARGV[1]|") or die "Unable to open $ARGV[1]: $!\n";
my $dataright = {};
my $nid=0;
while( <FIC> ) {
  my $line = $_;
  next if( $line =~ /^#/ );
  my @v = split( "\t", $line);

  if( defined( ${dataleft}->{"$v[2]$v[3]"} ) ) {
    my $d = $nid - ${dataleft}->{"$v[2]$v[3]"};
    $d = -$d if( $d < 0 );
    $distance += $d;
    print LOC "$v[2] $v[3] $d\n";
  } else {
    print STDERR "Fatal error: $ARGV[1] does not have a $v[2] $v[3] operation. Traces are not comparable\n";
  }

  $nid++;
}
close(FIC);
close(LOC);

print "Distance between $ARGV[0] and $ARGV[1] is $distance\n";

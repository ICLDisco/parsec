#!perl -W
use strict;
use Getopt::Long;

my $nodefmt="<%R/%V/%T> %K(%P)<%p>";
my $nodeshapeexpr="0";
my $nodefcexpr="%k";
my $nodelcexpr="%r*%V*%T+%v*%T+%t";
my $linkfmt="%S=>%D";
my $linkcolorexpr = "%E ? \"00FF00\" : \"FF0000\"";
my $linkstyleexpr = "%C ? \"dashed\" : \"solid\"";

my @shapes=("ellipse", "circle", "oval", "box", "polygon", "diamond", "pentagon", "hexagon", "septagon", "octagon", "square", "house", "invhouse", "trapezium","invtrapezium");
my @colors=("589AB7", "D7ACEF", "ABD6FD", "D2FC3D", "993300","9933FF","CC3300", "CC33FF", "FF3300","FF33FF","FFFF00");
my $ignore={};
my $inputs=[];

sub parseArgs {
  my $help = 0;
  my @il;
  my $result = GetOptions ("nodefmt=s"   => \$nodefmt,
                           "nodeshape=s" => \$nodeshapeexpr,
                           "nodefc=s"    => \$nodefcexpr,
                           "nodelc=s"    => \$nodelcexpr,
                           "linkfmt=s"   => \$linkfmt,
                           "linkc=s"     => \$linkcolorexpr,
                           "links=s"     => \$linkstyleexpr,
                           "help!"       => \$help,
                           "ignore=s"    => \@il,
                           "input=s"     => \@{$inputs});
  foreach my $f (@ARGV) {
    push @{$inputs}, $f;
  }

  if( $help ) {
    print STDERR <<END;
usage:
   --nodefmt             Define the text in the node.
                           Expect a printf-like format.
                           Can use %R, %V, %T, %K, %P, %L, and %p (see below)
                           Default: '$nodefmt'
   --nodeshape           Define the shape of the node.
                           Expect an integer expression.
                           Can use %k, %r, %v, %t, %R, %V, %T (see below)
                           Default: '$nodeshapeexpr'
                           Shapes are:

END
    my $r=0;
    foreach my $s (@shapes) {
      print STDERR "                             $r => $s\n";
      $r++;
    }
    print STDERR <<END;
   --nodegfc             Define the fill color of the node.
                           Expect an integer expression.
                           Can use %k, %r, %v, %t, %R, %V, %T (see below)
                           Default: '$nodefcexpr'
                           See palette below.
   --nodelc              Define the line color of the node.
                           Expect an integer expression.
                           Can use %k, %r, %v, %t, %R, %V, %T (see below)
                           Default: '$nodelcexpr'
                           See palette below.
   --linkfmt             Define the text on the link.
                           Expect a printf-like format.
                           Can use %S, %D, %s and %d (see below)
                           Default: '$linkfmt'
   --linkc               Define the color of the link.
                           Expect an RGB color (without the # sign)
                           Can use %E and %C (see below)
                           Default: '$linkcolorexpr'
   --links               Define the line style of the node.
                           Expect a string expression that reduces to "solid" or "dashed"
                           Can use %E and %C (see below)
                           Default: '$linkstyleexpr'
   --ignore=KERNEL      Ignore this kernel. This option can appear multiple times
   --input=INPUT        Add this input file. This option can appear multiple times.
                        Remaining arguments (unparsed) are considered as other input files.
                        The ordering of input files define the ranks.

   -------------
   Variables:
   -------------
    node format
     %R  Rank that ran the task
     %V  ID of the virtual process that ran the task
     %T  ID of the thread that ran the task
     %K  Name of the kernel
     %P  Parameters of the kernel
     %L  Locals of the kernel (Parameters included)
     %p  priority of the task
    node shape, fill color and nodecolor
     %k  id of the kernel (from 0 to max-kernel-index)
     %r  rank that ran the task
     %v  virtual process id that ran the tas
     %t  thread id that ran the task
     %R  number of processes
     %V  number of virtual processes per rank
     %T  number of threads per virtual processes
    link format
     %D  Destination Variable Name
     %S  Source Variable Name
     %d  Destination Rank
     %s  Source Rank
    link style or color
     %E  The link was an enabled link (boolean)
     %C  The link was a communicating link (boolean)
   ------------
   Colors:
   ------------
END
    $r=0;
    foreach my $s (@colors) {
      print STDERR "                             $r => #$s\n";
      $r++;
    }
    exit 0;
  }

  %{$ignore} = map { $_ => 1 } @il;
}

parseArgs(@ARGV);

my $KERNELS = {};
my $TASKS = {};

$nodeshapeexpr =~ s/%T/\$NT/g;
$nodeshapeexpr =~ s/%V/\$NV/g;
$nodeshapeexpr =~ s/%R/\$NR/g;
$nodeshapeexpr =~ s/%t/\$T/g;
$nodeshapeexpr =~ s/%v/\$V/g;
$nodeshapeexpr =~ s/%r/\$R/g;
$nodeshapeexpr =~ s/%k/\$K/g;

$nodefcexpr =~ s/%T/\$NT/g;
$nodefcexpr =~ s/%V/\$NV/g;
$nodefcexpr =~ s/%R/\$NR/g;
$nodefcexpr =~ s/%t/\$T/g;
$nodefcexpr =~ s/%v/\$V/g;
$nodefcexpr =~ s/%r/\$R/g;
$nodefcexpr =~ s/%k/\$K/g;

$nodelcexpr =~ s/%T/\$NT/g;
$nodelcexpr =~ s/%V/\$NV/g;
$nodelcexpr =~ s/%R/\$NR/g;
$nodelcexpr =~ s/%t/\$T/g;
$nodelcexpr =~ s/%v/\$V/g;
$nodelcexpr =~ s/%r/\$R/g;
$nodelcexpr =~ s/%k/\$K/g;

$linkcolorexpr =~ s/%E/\$EL/g;
$linkcolorexpr =~ s/%C/\$CL/g;

$linkstyleexpr =~ s/%E/\$EL/g;
$linkstyleexpr =~ s/%C/\$CL/g;

my $NT=0;
my $NV=0;
my $NR=0;

sub kernelID {
  my $K=shift;
  return $KERNELS->{$K} if( defined( $KERNELS->{$K} ) );
  $KERNELS->{$K} = keys(%$KERNELS);
  return $KERNELS->{$K};
}

sub linkColor {
  my ($EL, $CL) = @_;
  my $value = eval "$linkcolorexpr";
  return $value;
}

sub linkStyle {
  my ($EL, $CL) = @_;
  my $value = eval "$linkstyleexpr";
  return $value;
}

sub nodeShape {
  my ($R, $V, $T, $K) = @_;
  my $value = eval "$nodeshapeexpr";
  return $shapes[ $value % $#shapes ];
}

sub nodeFillColor {
  my ($R, $V, $T, $K) = @_;
  my $value = eval "$nodefcexpr";
  return $colors[ $value % $#colors ];
}

sub nodeLineColor {
  my ($R, $V, $T, $K) = @_;
  my $value = eval "$nodelcexpr";
  return $colors[ $value % $#colors ];
}

sub outputNode {
  my ($ID, $R, $V, $T, $K, $P, $op, $p) = @_;

  my $label = $nodefmt;
  $label =~ s/%R/$R/g;
  $label =~ s/%T/$T/g;
  $label =~ s/%V/$V/g;
  $label =~ s/%K/$K/g;
  $label =~ s/%P/$P/g;
  $label =~ s/%L/$op/g;
  $label =~ s/%p/$p/g;

  my $Kid = kernelID($K);
  my $nodeshape = nodeShape($R, $V, $T, $Kid);
  my $nodefill  = nodeFillColor($R, $V, $T, $Kid);
  my $nodeline  = nodeLineColor($R, $V, $T, $Kid);

  print "$ID [pencolor=\"#$nodeline\",shape=\"$nodeshape\",style=filled,fillcolor=\"#$nodefill\",fontcolor=\"black\",label=\"$label\"];\n"
}

sub nodeRank {
  my ($ID) = @_;
  return $TASKS->{$ID}->{R};
}

sub computeSpaceNode {
  my ($ID, $R, $V, $T, $K, $P, $op, $p) = @_;

  $NR = $R+1 if( $R + 1 > $NR );
  $NV = $V+1 if( $V + 1 > $NV );
  $NT = $T+1 if( $T + 1 > $NT );
  $TASKS->{$ID} = {};
  $TASKS->{$ID}->{'R'} = $R;
  $TASKS->{$ID}->{'V'} = $V;
  $TASKS->{$ID}->{'T'} = $T;
}

sub ignored {
  my ($k) = @_;
  foreach my $y ( keys %{$ignore} ) {
    if( $k =~ /$y/ ) {
      return 1;
    }
  }
  return 0;
}

sub onNodes {
  my $fct = shift;
  my @argv = @_;

  my $R=0;
  foreach my $f (@argv) {
    open(F, "<", $f) or die "Unable to open $f: $!\n";
    my $lnb=0;
    while (<F>) {
      my $line=$_;
      $lnb++;
      next if ($line =~ /^digraph G \{$/);
      last if ($line =~ /^\}/);
      next if ($line =~ / -> /);
      my ($ID, $COLOR, $T, $V, $K, $P, $op, $p);
      if( ($ID, $COLOR, $T, $V, $K, $P, $op, $p) = ($line =~ /^([^ ]+) \[shape="[^"]+",style=filled,fillcolor="#(......)",fontcolor="black",label="<([0-9]+)\/([0-9]+)> ([^(]+)\(([^\)]*)\)\[([^>]*)\]<([^>]+)>/) ) {
        if( !ignored($K) ) {
          $fct->($ID, $R, $V, $T, $K, $P, $op, $p);
        }
      } else {
        print STDERR "  Error on $f:$lnb malformed line $line\n";
      }
    }
    $R++;
    close(F);
  }
}

sub outputLink {
  my ($ID1, $ID2, $VSRC, $VDST, $NSRC, $NDST, $EL) = @_;
  my $label = $linkfmt;

  $label =~ s/%S/$VSRC/g;
  $label =~ s/%D/$VDST/g;
  $label =~ s/%s/$NSRC/g;
  $label =~ s/%d/$NDST/g;

  my $same = 0;
  if( ref($NSRC) eq "SCALAR" ) {
    if( ref($NDST) eq "SCALAR" ) {
      $same = ($NSRC == $NDST);
    } else {
      $same = ($NSRC == ($NDST + 0));
    }
  } else {
    if( ref($NDST) eq "SCALAR" ) {
      $same = (($NSRC + 0) == $NDST);
    } else {
      $same = ( $NSRC eq $NDST );
    }
  }

  my $color=linkColor($EL, !$same);
  my $style=linkStyle($EL, !$same);

  print "$ID1 -> $ID2 [label=\"$label\" color=\"#$color\" style=\"$style\"];\n";
}

sub onLinks {
  my $fct = shift;
  my @argv = @_;

  my $R=0;
  foreach my $f (@argv) {
    open(F, "<", $f) or die "Unable to open $f: $!\n";
    my $lnb=0;
    while (<F>) {
      my $line=$_;
      $lnb++;
      next if ($line =~ /^digraph G \{$/);
      last if ($line =~ /^\}/);
      next unless ($line =~ / -> /);
      my ($ID1, $ID2, $VSRC, $VDST, $COLOR, $NSRC, $NDST);
      if( ($ID1, $ID2, $VSRC, $VDST, $COLOR) = ($line =~ /^([^ ]+) -> ([^ ]+) \[label="([^=]+)=>([^"]+)",color="#(......)"/) ) {
        if( exists($TASKS->{$ID1}) ) {
          $NSRC=nodeRank($ID1);
	} else {
	  $NSRC="Unknown";
	}
	if( exists($TASKS->{$ID2}) ) {
          $NDST=nodeRank($ID2);
	} else {
	  $NDST="Unknown";
	}
	my $EL=( $COLOR eq "00FF00" );
	$fct->($ID1, $ID2, $VSRC, $VDST, $NSRC, $NDST, $EL);
      } else {
        print STDERR "  Error on $f:$lnb malformed line $line\n";
      }
    }
    $R++;
    close(F);
  }
}

print "digraph G {\n";
onNodes(\&computeSpaceNode, @{$inputs});
onNodes(\&outputNode, @{$inputs});
onLinks(\&outputLink, @{$inputs});
print "}\n";

#!/usr/bin/perl -w

use Data::Dumper;

#my ($lineok, $linefound);

sub parse_access {
  my ($MEM) = @_;
  my @acc = split /,/, $MEM;
  my $t = {};

  foreach my $a (@acc) {
    if( my ($type, $memref) = ($a =~ /^([RWM])(.*)$/) ) {
      if( defined $t->{ $memref } ) {
	next if( $t->{ $memref } eq $type );
	next if( $t->{ $memref } eq "M" );
	next if( $t->{ $memref } eq "W" );
	$t->{ $memref } = $type;
      } else {
	$t->{ $memref } = $type;
      }
    } else {
      die "Malformed memory reference: $a\n";
    }
  }

  my @r;
  foreach my $memref (keys %{$t}) {
    push @r, "" . $t->{$memref} . $memref;
    delete $t->{$memref};
  }

  return @r;
}

while ( <> ) {
  my $line = $_;
  my $warncleanup = 0;
#  $linefound++;
  chomp $line;
  my $data = {};
  my @exec = split / /, $line;
  foreach my $op (@exec) {
    my ($SE, $T, $MEM);
    if ( ($SE, $T, $MEM) = ($op =~ /^([SE])\#([^\#]+)\#([^\#]*)\#/) ) {
      if ( $SE eq "S" ) {
	my @accesses = parse_access($MEM);
	foreach my $access (@accesses) {
	  my ($type,$memref);
	  if ( ($type, $memref) = ($access =~ /^([RWM])(.*)$/) ) {
#	    print "Considering access $type on $memref by task $T\n";
	    if ( defined($data->{$memref}) &&
	         defined($data->{$memref}->{writer}) && 
		 defined($data->{$memref}->{readers}) ) {
	      if ( length($data->{$memref}->{writer}) > 0 ) {
		my $error;
		if ( $type eq "M" || $type eq "W" ) {
		  $error = "Write/Write";
		} else {
		  $error = "Write/Read";
		}
		die "Found a $error scenario related to task $T: task " . $data->{$memref}->{writer} . " can write in parallel.\n$line\n";
		goto CLEANUP;
	      } elsif ( @{$data->{$memref}->{readers}} > 0 ) {
		if ( $type eq "R" ) {
		  push @{$data->{$memref}->{readers}}, $T;
#		  print "Adding $T to the readers of $memref " . Dumper( $data->{$memref} ) . "\n";
		} else {
		  my $error = join(", ", @{$data->{$memref}->{readers}});
		  die "Found a Write/Read scenario related to task $T: task $T can write while some tasks ($error) can read in parallel\n$line\n";
		  goto CLEANUP;
		}
	      } else {
		$data->{$memref} = {};
		$data->{$memref}->{writer} = "";
		$data->{$memref}->{readers} = [];
		if ( $type eq "M" || $type eq "W" ) {
		  $data->{$memref}->{writer} = $T;
#		  print "Setting $T as the writer of $memref " . Dumper( $data->{$memref} ) . "\n";
		} else {
		  push @{$data->{$memref}->{readers}}, $T;
#		  print "Adding $T to the readers of $memref " . Dumper( $data->{$memref} ) . "\n";
		}
	      }
	    } else {
	      if( defined( $data->{$memref} ) ) {
#		print "Cleaning up $memref " . Dumper( $data->{$memref} ) . "\n";
		delete $data->{$memref}->{writer}  if( defined($data->{$memref}->{writer}) );
		delete $data->{$memref}->{readers} if( defined($data->{$memref}->{readers}) );
		delete $data->{$memref};
	      }

	      $data->{$memref} = {};
	      $data->{$memref}->{writer} = "";
	      $data->{$memref}->{readers} = [];
	      if ( $type eq "M" || $type eq "W" ) {
		$data->{$memref}->{writer} = $T;
#		print "Setting $T as the writer of $memref " . Dumper( $data->{$memref} ) . "\n";
	      } else {
		push @{$data->{$memref}->{readers}}, $T;
#		print "Adding $T to the readers of $memref " . Dumper( $data->{$memref} ) . "\n";
	      }
	    }
	  } else {
	    die "malformed input: $op (memory reference $memref does not start with R, W or M)\n";
	  }
	}
      } else {
	die "malformed input: $op ($SE does not start with S or E)\n" unless ($SE eq "E");
	my @accesses = parse_access($MEM);
	foreach my $access (@accesses) {
	  my $memref;
	  if ( ($memref) = ($access =~ /^[RWM](.*)$/) ) {
#	    print "Considering end of access on $memref by task $T\n";
	    if ( defined($data->{$memref}->{writer}) &&
		 $data->{$memref}->{writer} eq "$T" ) {
	      $data->{$memref}->{writer} = "";
#	      print "Removing $T as the writer of $memref " . Dumper( $data->{$memref} ) . "\n";
	      if ( !defined($data->{$memref}->{readers}) ||
		   @{$data->{$memref}->{readers}} == 0 ) {
#		print "Removing $memref\n";
		delete $data->{$memref}->{readers};
		delete $data->{$memref}->{writer};
		delete $data->{$memref};
	      }
	    } else {
	      my $idx;
	      if( defined( $data->{$memref}->{readers} ) ) {
		for ($idx = 0; $idx < @{$data->{$memref}->{readers}}; $idx++) {
		  if ( @{$data->{$memref}->{readers}}[$idx] eq "$T" ) {
		    last;
		  }
		}
		splice(@{$data->{$memref}->{readers}}, $idx, 1);
#		print "Removing $T as one of the readers of $memref " . Dumper( $data->{$memref} ) . "\n";
		if ( @{$data->{$memref}->{readers}} == 0 ) {
		  if ( !defined $data->{$memref}->{writer} ||
		       $data->{$memref}->{writer} eq "" ) {
		    delete $data->{$memref}->{readers};
		    delete $data->{$memref}->{writer};
		    delete $data->{$memref};
#		    print "Removing $memref\n";
		  }
		}
	      }
	    }
	  } else {
	    die "malformed input: $op (memory reference $memref does not start with R, W or M)\n";
	  }
	}
      }
    } else {
      die "Malformed input: $op does not match the general pattern\n";
    }
  }

#  $lineok++;
#  print "OK: " . $lineok . "/" . $linefound . "( = " .(100.0 * $lineok)/$linefound . "%)\r";

  $warncleanup = 1;
 CLEANUP:
  my $foundwarn = 0;
  while (my ($memref, $rec) = each (%{$data})) {
    if( $warncleanup ) {
      if( ! ($rec->{writer} eq "") ) {
	print "Warning: there is still a writer (task " . $rec->{writer} . ") on memory $memref at the end of that execution\n";
	$warn=1;
      }
      if ( 0 != @{$rec->{readers}} ) {
	my $readers = join(', ', @{$rec->{readers}});
	print "Warning: there are still readers (tasks $readers) on memory $memref at the end of that execution\n";
	$warn=1;
      }
    }
    delete $data->{$memref}->{writer};
    delete $data->{$memref}->{readers};
    delete $data->{$memref};
  }
  if( $foundwarn ) {
    print "$line\n";
  }
}

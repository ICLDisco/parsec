#!/usr/bin/perl -w

my $data = {};

while ( <> ) {
  my $line = $_;
  chomp $line;
  my @exec = split / /, $line;
  foreach my $op (@exec) {
    my ($SE, $T, $MEM);
    if ( ($SE, $T, $MEM) = ($op =~ /^([SE])\#([^\#]+)\#([^\#]*)\#/) ) {
      if ( $SE eq "S" ) {
	my @accesses = split /,/, $MEM;
	foreach my $access (@accesses) {
	  my ($type,$memref);
	  if ( ($type, $memref) = ($access =~ /^([RWM])(.*)$/) ) {
	    print "Considering access $type on $memref by task $T\n";
	    if ( defined $data->{$memref} ) {
	      if ( ! ($data->{$memref}->{writer} eq "") ) {
		my $error;
		if ( $type eq "M" || $type eq "W" ) {
		  $error = "Write/Write";
		} else {
		  $error = "Write/Read";
		}
		die "Found a $error scenario related to task $T: task " . $data->{$memref}->{writer} . " can write in parallel.\n$line\n";
	      } elsif ( @{$data->{$memref}->{readers}} > 0 ) {
		if ( $type eq "R" ) {
		  print "Adding $T to the readers of $memref\n";
		  push @{$data->{$memref}->{readers}}, $T;
		} else {
		  my $error = join(", ", @{$data->{$memref}->{readers}});
		  die "Found a Write/Read scenario related to task $T: task $T can write while some tasks ($error) can read in parallel\n$line\n";
		}
	      
	      } else {
		$data->{$memref} = {};
		$data->{$memref}->{writer} = "";
		$data->{$memref}->{readers} = [];
		if ( $type eq "M" || $type eq "W" ) {
		  print "Setting $T as the writer of $memref\n";
		  $data->{$memref}->{writer} = $T;
		} else {
		  print "Adding $T to the readers of $memref\n";
		  push @{$data->{$memref}->{readers}}, $T;
		}
	      }
	    } else {
	      $data->{$memref} = {};
	      $data->{$memref}->{writer} = "";
	      $data->{$memref}->{readers} = [];
	      if ( $type eq "M" || $type eq "W" ) {
		print "Setting $T as the writer of $memref\n";
		$data->{$memref}->{writer} = $T;
	      } else {
		print "Adding $T to the readers of $memref\n";
		push @{$data->{$memref}->{readers}}, $T;
	      }
	    }
	  } else {
	    die "malformed input: $op (memory reference $memref does not start with R, W or M)\n";
	  }
	}
      } else {
	die "malformed input: $op ($SE does not start with S or E)\n" unless ($SE eq "E");
	my @accesses = split /,/, $MEM;
	foreach my $access (@accesses) {
	  my $memref;
	  if ( ($memref) = ($access =~ /^[RWM](.*)$/) ) {
	    print "Considering end of access on $memref by task $T\n";
	    if ( $data->{$memref}->{writer} eq "$T" ) {
	      print "Removing $T as the writer of $memref\n";
	      delete $data->{$memref}->{writer};
	      my $nbreaders = @{$data->{$memref}->{readers}};
	      if ( $nbreaders == 0 ) {
		print "Removing $memref\n";
		delete $data->{$memref};
	      }
	    } else {
	      my $idx;
	      for ($idx = 0; $idx < @{$data->{$memref}->{readers}}; $idx++) {
		if ( @{$data->{$memref}->{readers}}[$idx] eq "$T" ) {
		  last;
		}
	      }
	      splice(@{$data->{$memref}->{readers}}, $idx, 1);
	      print "Removing $T as one of the readers of $memref\n";
	      if ( @{$data->{$memref}->{readers}} == 0 ) {
		if ( !defined $data->{$memref}->{writer} ) {
		  delete $data->{$memref};
		  print "Removing $memref\n";
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
}

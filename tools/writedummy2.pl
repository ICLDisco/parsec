#!perl -W
use strict;
use Getopt::Long;
use POSIX;

my %nodes = ();

sub getNodes {
    
    my ($filename) = @_;
    
    open FILE, $filename or die $!;
    
    while (<FILE>) {
        my $line = $_;
        chop $line;
        
        my $nodename = $line;
        $nodename =~ s/(.*) (.*)/$1/;
        my $nodeprop = $2;
        
        #print $nodename."\n";
        #print $nodeprop."\n";
        
        $nodes{ $nodename } = {  'offname' => 0,
                                 'offprop' => 0,
                                 'nbsucc' => 0,
                                 'prop' => $nodeprop,
                                 'succ' => [] };
    }
    
    close FILE;
}


sub addDependencies {
    my ($filename) = @_;

    my $tmp = "/tmp/writedummy.tmp";
    my $tmp2 = "/tmp/writedummy2.tmp";
    system( "grep '^.* -> .* \\[.*' $filename > $tmp" );
    system( "sed -i 's/\\(.* -> .*\\) \\[.*/\\1/' $tmp" );
    system( "sort -u $tmp > $tmp2" );

    open FILE, $tmp2 or die $!;
    while (<FILE>) {
        my $line = $_;
        chop $line;
        
        if ( $line =~ /(.*) -> (.*) \[.*/ ) {
            my $node1 = $1;
            my $node2 = $2;
            if ( !(exists $nodes{$node1}) ) {
                print $node2." is not in this DAG\n";
                next;
            }
            if ( !(exists $nodes{$node2}) ) {
                print $node2." is not in this DAG\n";
                next;
            }

            $nodes{$node1}{'nbsucc'}++;
            push( @{$nodes{$node1}{'succ'}}, $node2 );
        }
    }
    close FILE;

    system("rm -f $tmp $tmp2");

}

sub printNodes {
    # Write filenode_header_t 
    foreach my $key (keys %nodes) {
        print "$key: $nodes{$key}\n";
    }
}

sub getNodeIndex {
    my ($key) =  @_;
    my $i = 0;
    
    foreach my $key2 (keys %nodes)
    {
        if ( $key eq $key2 ) {
            return $i;
        } else {
            $i++;
        }
    }
    return -1;
}

sub writeNodes {
    
    my ($filename) = @_;
    my $filenode_header_size = 104;
    my $filenode_size = 32;
    my $nbnodes = keys %nodes;
    my $nextstring = $filenode_header_size;

    open(FILE, '>:raw', $filename );

    print FILE pack('i', $nbnodes);
    foreach my $key (keys %nodes) {
        print FILE $key;
        print FILE pack('c', 0);
        print FILE $nodes{$key}{'prop'};
        print FILE pack('c', 0);
        print FILE pack('i', $nodes{$key}{'nbsucc'});

        my $i=0;
        for(; $i<$nodes{$key}{'nbsucc'}; $i++) {
            my $succ = $nodes{$key}{'succ'}[$i];
            print FILE pack( 'i', getNodeIndex( $succ ) );
        }
    }

    my $offset = tell FILE;
    my $pagesize = POSIX::sysconf(POSIX::_SC_PAGESIZE);

    for(my $i = 0; (($i + $offset) % $pagesize) != 0; $i++) {
        print FILE pack('c', 0);
    }
    
    close FILE;
}


getNodes( $ARGV[0].".txt" );
addDependencies( $ARGV[0].".dot" );
#printNodes();
writeNodes( $ARGV[0].".out" );

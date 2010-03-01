CXX=g++-4.2
CXXFLAGS=-O3
LDFLAGS=-L/opt/local/lib -lboost_regex-mt

all:	kill

s2p:	smpss_to_petit.cpp
	$(CXX) $(CXXFLAGS) smpss_to_petit.cpp -o smpss_to_petit -I/opt/local/include $(LDFLAGS)

kill:	kill.cpp
	$(CXX) $(CXXFLAGS) kill.cpp -o kill

clean:
	@rm -rf *.o *.dSYM kill

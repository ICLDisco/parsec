#!/bin/sh

in=$1
out=$2

process() {
    cp $in $out
    while read nodeid name; do
	sed 's/"'$nodeid'"/"'$name'"/g' $out | grep -v "xlink.*\"$name\"" |grep -v '^</a>$' > $out.tmp
	mv -f $out.tmp $out
    done
}

grep -2 "g id=\"node.*\" " $in | awk -v FS='"' '{ if(NR%3 < 2) printf("%s ", $2); else if(NR%3==2) printf("\n");}'  | grep node | process

head -n 8 $out | sed 's/svg width=/svg onload="startup(evt)" width=/g'  > $out.tmp
cat<<EOF>>$out.tmp
  <script><![CDATA[
var svgDoc;
var Root;
var cursel;
var scale=1;
var translatex=0;
var translatey=0;
function unselectAll() {
  var allG = Root.getElementsByTagName('g');
  for(i = 0; i < allG.length; i++) {
    if( allG.item(i).className.baseVal == "node" ) {
        allG.item(i).setAttribute("opacity", "0.5");
    }
  }
}
function startup(evt){
  O=evt.target
  svgDoc=O.ownerDocument;
  Root=svgDoc.documentElement;
  O.setAttribute("onmousedown","recolor(evt)")
  cursel = undefined;
  oldFill = "";
  unselectAll();
  top.svgzoom = svgzoom
  top.svgtranslatex = svgtranslatex
  top.svgtranslatey = svgtranslatey
  top.svg_outside_select = outsideSelect;

  svgDoc.getElementById('vp').setAttribute("transform", "translate(" +translatex + ", " + translatey + ") scale(" + scale + ")");

  top.ready()
}
function svgzoom( x ) {
  scale=x;
  svgDoc.getElementById('vp').setAttribute("transform", "translate(" + translatex + ", " + translatey + ") scale(" + scale + ")");
}
function svgtranslatex( x ) {
  translatex=-x;
  svgDoc.getElementById('vp').setAttribute("transform", "translate(" + translatex + ", " + translatey + ") scale(" + scale + ")");
}
function svgtranslatey( x ) {
  translatey=-x;
  svgDoc.getElementById('vp').setAttribute("transform", "translate(" + translatex + ", " + translatey + ") scale(" + scale + ")");
}
function recolor(evt){
  if( cursel != undefined ) {
      cursel.setAttribute("opacity", "0.5");
      cursel = undefined;
  }

  if( evt.target.parentNode.className.baseVal == "node"  ) {
      cursel = evt.target.parentNode;
      cursel.setAttribute("opacity", "1");

      top.select_function( evt.target.parentNode.id );
  } else {
      top.select_function( undefined );
  }
}
function outsideSelect(x){
  if( cursel != undefined ) {
      cursel.setAttribute("opacity", "0.5");
      cursel = undefined;
  }

  cursel = svgDoc.getElementById(x);
  if( !cursel ) {
    opera.postError("graph.svg warning: unable to find the element named " + x);
  } else {
    cursel.setAttribute("opacity", "1");
  }
}
//]]>
  </script>
EOF
nb=$(wc -l $out | awk '{print $1}')
tail -n $(($nb - 8)) $out | sed 's/<title>G<\/title>/<title>G<\/title><g id="vp">/g' | sed 's/<\/svg>/<\/g><\/svg>/g' >> $out.tmp
mv -f $out.tmp $out

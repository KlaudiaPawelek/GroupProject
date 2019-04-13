<style type ="text/css">

    .field__label{
		display: none;
	}
	
	.slide{
		display: none;
		padding - top: 10 px;
		border-radius: 15px;
	}
	
	.image{
		padding - top: 100 px;
		border-radius: 15px; 
		width: 100%;
		height: 80%;
		
	}
	
	.text, .average{
		border-radius: 15px;
		text-align: center;
		font-size: 15px
		font-weight: bold;
		padding: 8px 8px;	
        color: #ffffff;
        background: rgba(128,128,128,0.8);
	}
	
	.prev, .next{
		
		position: absolute;
		top: 400px;
		color: #f4473a;
		font-weight: bold;
		padding: 5px 15px;
		font-size:50px;
		border-radius: 0 3px 3px 0;
	}
	
	.next{
	    border-radius: 3px 0 0 3px;
		right: 0;
	}
	
	.prev:hover, .next:hover  {
		background: rgba(0,0,0,0.8);
	}
	
</style>

<body>


<div  id="container">

</div>

<div id="arrow">
	<a class="prev"  onclick="next(-1)">&lt;</a>
	<a class="next"  onclick="next(+1)">&gt;</a>
</div>
</body>

<?php
	header('Content-type: text/javascript');
	$dir = 'sites/default/files/pictures/output/';
    $images = array();
    $index = 0;
    if ($opendir = opendir($dir)) {
        while (($file = readdir($opendir)) !== FALSE){
            if($file !== "." && $file !== "..") {
				$ext = pathinfo($file, PATHINFO_EXTENSION);
				if( $ext == 'gif' || $ext == 'png' || $ext == 'jpg' ) {
					$src= $dir.$file;
					array_push($images, $src);
				}
            }
        }
    }
	
	$infos = array();
	$info = array("","","","");
	$datas=file_get_contents("sites/default/files/pictures/output/outputFile.txt");
	$splitcontents = explode("Name of image:", $datas);
	foreach($splitcontents as $image){
		$image = explode("\n", $image);
		for ($i=1;$i<count($image);$i++){
			$temp=explode (":",$image[$i]);
			$info[$i-1]=$temp[1];
		}
		array_push($infos, $info);
	}
	
	
	echo"<script >
	
  var index = 1;
  
  upload();
  showImage(index);
  
	
function next(n){
	index = index + n ;
	showImage(index); 
}

function upload(){
	var divAvg = document.createElement('DIV');
	divAvg.setAttribute('class', 'average');
	var PAvg =document.createElement('P');
		var br = document.createElement('br');
		var textAvg0 = document.createTextNode('The average result from all the image are the following: ');
	PAvg.appendChild(textAvg0);
	PAvg.appendChild(br);
		var br = document.createElement('br');
		var jolie = 'Nb bubbles all : '.concat(".$infos[count($images)][6].");
		var textAvg1 = document.createTextNode(jolie);
	PAvg.appendChild(textAvg1);
	PAvg.appendChild(br);
		var br = document.createElement('br');
		var jolie = 'SMD all : '.concat(".$infos[count($images)][7].");
		var textAvg2 = document.createTextNode(jolie);
	PAvg.appendChild(textAvg2);
	PAvg.appendChild(br);
		var br = document.createElement('br');
		var jolie = 'Mean Diameter all : '.concat(".$infos[count($images)][8].");
		var textAvg3 = document.createTextNode(jolie);
	PAvg.appendChild(textAvg3);
	PAvg.appendChild(br);
		var br = document.createElement('br');
		var jolie = 'Min Diameter all : '.concat(".$infos[count($images)][9].");
		var textAvg4 = document.createTextNode(jolie);
	PAvg.appendChild(textAvg4);
	PAvg.appendChild(br);
		var br = document.createElement('br');
		var jolie = 'Max Diameter all : '.concat(".$infos[count($images)][10].");
		var textAvg5 = document.createTextNode(jolie);
	PAvg.appendChild(textAvg5);
	PAvg.appendChild(br);
		var br = document.createElement('br');
		var jolie = 'Time : '.concat(".$infos[count($images)][11].");
		var jolie2 = jolie.concat(' sec');
		var textAvg6 = document.createTextNode(jolie);
	PAvg.appendChild(textAvg6);
	divAvg.appendChild(PAvg);
	document.getElementById('container').appendChild(divAvg);
	
	
	";
	
	
	for ($i=0; $i<count($images);$i++){
		
	echo"
		
	
	    var divP".$i." =document.createElement('DIV');
		divP".$i.".setAttribute('class', 'text');
		var P".$i." =document.createElement('P');
		var br = document.createElement('br');
		var jolie = 'Nb bubbles: '.concat(".$infos[$i+1][0].");
		var textA".$i." = document.createTextNode(jolie);
		P".$i.".appendChild(textA".$i.");
		P".$i.".appendChild(br);
		var br = document.createElement('br');
		var jolie = 'SMD: '.concat(".$infos[$i+1][1].");
		var textB".$i." = document.createTextNode(jolie);
                P".$i.".appendChild(textB".$i.");
		P".$i.".appendChild(br);
		var br = document.createElement('br');
		var jolie = 'Mean Diametre: '.concat(".$infos[$i+1][2].");
		var textC".$i." = document.createTextNode(jolie); 
	        P".$i.".appendChild(textC".$i.");
		P".$i.".appendChild(br);
		var br = document.createElement('br');
		var jolie = 'Min Diametre: '.concat(".$infos[$i+1][3].");
		var textD".$i." = document.createTextNode(jolie);
		P".$i.".appendChild(textD".$i.");
                P".$i.".appendChild(br);
		var br = document.createElement('br');
		var jolie = 'Max Diametre: : '.concat(".$infos[$i+1][4].");
		var textE".$i." = document.createTextNode(jolie);
		P".$i.".appendChild(textE".$i.");
		P".$i.".appendChild(br);
		divP".$i.".appendChild(P".$i.");
	
	
		var image".$i." = document.createElement('IMG');
		image".$i.".setAttribute('src', '" .$images[$i]."');	
		image".$i.".setAttribute('class', 'image');
		var slide".$i." = document.createElement('DIV');
		slide".$i.".setAttribute('class','slide');
		slide".$i.".appendChild(divAvg".$i.");
		slide".$i.".appendChild(image".$i.");
		slide".$i.".appendChild(divP".$i.");		
		document.getElementById('container').appendChild(slide".$i.");
		
		
	";}
	
	
	if(count($images)==0){
		
		echo "
		var imageNo = document.createElement('IMG');
		imageNo.setAttribute('src', 'sites/default/files/pictures/noimage.jpeg');	
		imageNo.setAttribute('class', 'image');
		var slideNo = document.createElement('DIV');
		slideNo.setAttribute('class','slide');
		slideNo.appendChild(imageNo);
		document.getElementById('container').appendChild(slideNo);
		
		";
	}
  

	
	echo "
		
	
}	

function showImage(n){
	var x = document.getElementsByClassName('slide');
	if (n>x.length){index=1};
	if (n<1){index=x.length};
	for (var i=0;i<x.length;i++){
		x[i].style.display = 'none';
	}
	x[index-1].style.display = 'block';
}

</script>";
?>
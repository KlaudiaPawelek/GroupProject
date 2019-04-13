<?php
              function gotoleft(&$index, $nbImages ){
                  $index=$index+1;
                  if ($index>$nbImages){$index=0;}
              }
              function gotoright(&$index, $nbImages){
                  $index=$index-1;
                  if ($index<0){$index=$nbImages;}
              }
              echo '<p>Your Results:</p>';
              $dir = "sites/default/files/pictures/output/";
              $images = array();
              $index = 0;
             if ($opendir = opendir($dir)) {
                  while (($file = readdir($opendir)) !== FALSE){
                      if($file !== "." && $file !== "..") {
                          array_push($images, $file);
                      }
                  }
              }
              $nbImages = count($images);
              echo "<img src='$dir/$images[$index]'> <br>";
              echo '<input type="button" name="left" id="left" value="<"  onclick="gotoleft($index,$nbImages)" />';
              echo '<input type="button" name="right" id="test" value=">"  onclick="<?php gotoright($index,$nbImages); ?>" /><br/>';
?>

<p>Your Result</p>

<p><img id="image" src="sites/default/files/pictures/output/img1.jpg" /></p>

<button onclick="gotoleft()">&lt;</button>
<button onclick="gotoright()">&gt;</button>

<script>

$(function(){
	var folder ="sites/default/files/pictures/output/"; 
	$.ajax({
		url : folder,
		success: function (data) {
			$(data).find("a").attr("href", function (i, val) {
				if( val.match(/\.(jpe?g|png|gif)$/) ) { 
					$("body").append( "<img src='"+ folder + val +"'>" );
				} 
			});
		}
	});
});

function gotoleft() {
  document.getElementById("image").src = "sites/default/files/pictures/output/img3.jpg";
}
function gotoright() {
  document.getElementById("image").src = "sites/default/files/pictures/output/img2.jpg";
}
</script>


<p>Your Result</p>

<p><img id="image" src="sites/default/files/pictures/output/img1.jpg" /></p>

<button onclick="gotoleft()">&lt;</button>
<button onclick="gotoright()">&gt;</button>

<ul>
    <?php
        $dirname = "sites/default/files/pictures/output/";
        $images = scandir($dirname);
        shuffle($images);
        $ignore = Array(".", "..");
        foreach($images as $curimg){
            if(!in_array($curimg, $ignore)) {
                echo "<a href='".$dirname.$curimg."'><img src='img.php?src=".$dirname.$curimg."&w=300&zc=1' alt='' /></a> ";
            }
        }                 
    ?>
</ul>

<script type="text/javascript">
$(function(){
	var folder ="sites/default/files/pictures/output/"; 
	$.ajax({
		url : folder,
		success: function (data) {
			$(data).find("a").attr("href", function (i, val) {
				if( val.match(/\.(jpe?g|png|gif)$/) ) { 
					$("body").append( "<img src='"+ folder + val +"'>" );
				} 
			});
		}
	});
});
function gotoleft() {
  document.getElementById("image").src = "sites/default/files/pictures/output/img3.jpg";
}
function gotoright() {
  document.getElementById("image").src = "sites/default/files/pictures/output/img2.jpg";
}
</script>

//document.getElementById('left').textContent= 'damien';

function images(){
	var folder ='sites/default/files/pictures/output/'; 
    var fs = require('fs');
    src = fs.readdirSync(folder);
    //document.getElementById('left').textContent= 'damien';
}	

<body>
<div id='container'>
</div>
<button onclick="myFunction()">Try it</button>
</body>
<?php
echo'<script>

  var x = document.createElement("IMG");
  x.setAttribute("src", "sites/default/files/pictures/output/img3.jpg");
  x.setAttribute("width", "304");
  x.setAttribute("height", "228");
  x.setAttribute("alt", "The Pulpit Rock");
  document.getElementById("container").appendChild(x);

</script>'
?>
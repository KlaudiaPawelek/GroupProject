<?php  
  $valid_formats = array("jpg", "png");
  $max_file_size = 1024*5000; //100 kb
  $path = "/var/www/html/drupal/sites/default/files/samples/"; /// Upload directory
  $count = 0;

  
  if(isset($_POST['upload']) and $_SERVER['REQUEST_METHOD'] == "POST")
  {
  
  // Loop $_FILES to exeicute all files
    foreach ($_FILES['files']['name'] as $f => $name) 
    {
      if ($_FILES['files']['error'][$f] == 4) 
      {
        continue; // Skip file if any error found
      }
      if ($_FILES['files']['error'][$f] == 0) 
      {
        if ($_FILES['files']['size'][$f] > $max_file_size) 
        {
          $message[] = "$name is too large!.";
          continue; // Skip large files
        }
        elseif( ! in_array(pathinfo($name, PATHINFO_EXTENSION), $valid_formats) )
        {
          $message[] = "$name is not a valid format";
          continue; // Skip invalid file formats
        }
        else
        { 
          // No error found! Move uploaded files
          if(move_uploaded_file($_FILES["files"]["tmp_name"][$f], $path.$name))
          {
            $count++; // Number of successfully uploaded file
          }  
        }
      }
    }
   
    header("Location: /drupal/images");
}
 
?>



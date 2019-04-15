<?php

  if(isset($_POST) and $_SERVER['REQUEST_METHOD'] == "POST")
  {
    shell_exec ("./bubble 0.2 /var/www/html/drupal/sites/default/files/pictures/output/ Haar5.xml /var/www/html/drupal/sites/default/files/samples/*jpg ");
    header("Location: /drupal/Results");
    
  }
  
?>
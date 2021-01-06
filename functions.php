<?php
    header('Content-Type: application/json');
    $response = array();
    //if(!isset($_POST['function'])) { $response['error'] = 'No function name!'; }

    //if(!isset($_POST['parameters'])) { $response['error'] = 'No function arguments!'; }

    if(!isset($response['error'])) {

        switch($_POST['function']) {
            case 'detect':
                $response["centers"] = shell_exec('python3 yolo_bing_cli.py ' . $_POST['parameters']);
                break;

            default:
               $response['error'] = 'Not found function '.$_POST['functionname'].'!';
               break;
        }

    }

    echo json_encode($response);
?>

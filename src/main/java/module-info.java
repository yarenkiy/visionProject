module com.example.visionproject {
    requires javafx.controls;
    requires javafx.fxml;


    opens com.example.visionproject to javafx.fxml;
    exports com.example.visionproject;
}
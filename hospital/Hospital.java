import java.util.*;

class Person {
    private String name;
    private int age;
    private String gender;
    private String contactNumber;

    public Person(String name, int age, String gender, String contactNumber) {
        this.name = name;
        this.age = age;
        this.gender = gender;
        this.contactNumber = contactNumber;
    }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public int getAge() { return age; }
    public void setAge(int age) { this.age = age; }

    public String getGender() { return gender; }
    public String getContactNumber() { return contactNumber; }
    public void setContactNumber(String contactNumber) { this.contactNumber = contactNumber; }
}

class Patient extends Person {
    private String address, medicalHistory, admissionDate, patientID;

    public Patient(String name, int age, String gender, String contactNumber, String address,
                   String medicalHistory, String admissionDate, String patientID) {
        super(name, age, gender, contactNumber);
        this.address = address;
        this.medicalHistory = medicalHistory;
        this.admissionDate = admissionDate;
        this.patientID = patientID;
    }

    public String getPatientID() { return patientID; }
    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }

    public String getMedicalHistory() { return medicalHistory; }
    public void setMedicalHistory(String medicalHistory) { this.medicalHistory = medicalHistory; }

    @Override
    public String toString() {
        return "Patient ID: " + patientID + ", Name: " + getName() + ", Age: " + getAge() +
               ", Gender: " + getGender() + ", Contact: " + getContactNumber() +
               ", Address: " + address + ", Medical History: " + medicalHistory +
               ", Admission Date: " + admissionDate;
    }
}

class Staff extends Person {
    private String role, department, staffID;

    public Staff(String name, int age, String gender, String contactNumber, String role, String department, String staffID) {
        super(name, age, gender, contactNumber);
        this.role = role;
        this.department = department;
        this.staffID = staffID;
    }

    public String getStaffID() { return staffID; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }

    public String getDepartment() { return department; }
    public void setDepartment(String department) { this.department = department; }

    @Override
    public String toString() {
        return "Staff ID: " + staffID + ", Name: " + getName() + ", Role: " + role +
               ", Department: " + department + ", Contact: " + getContactNumber();
    }
}

class Appointment {
    private String patientID, doctorID, dateTime, reason;

    public Appointment(String patientID, String doctorID, String dateTime, String reason) {
        this.patientID = patientID;
        this.doctorID = doctorID;
        this.dateTime = dateTime;
        this.reason = reason;
    }

    @Override
    public String toString() {
        return "Patient ID: " + patientID + ", Doctor ID: " + doctorID +
               ", Date/Time: " + dateTime + ", Reason: " + reason;
    }
}

// Class: Billing
class Bill {
    private String patientID;
    private double consultationFee, roomCharge, otherServices;
    private boolean paid;

    public Bill(String patientID, double consultationFee, double roomCharge, double otherServices) {
        this.patientID = patientID;
        this.consultationFee = consultationFee;
        this.roomCharge = roomCharge;
        this.otherServices = otherServices;
        this.paid = false;
    }

    public double calculateTotal() {
        return consultationFee + roomCharge + otherServices;
    }

    public void markAsPaid() {
        this.paid = true;
    }

    @Override
    public String toString() {
        return "Patient ID: " + patientID + ", Total Bill: " + calculateTotal() + ", Paid: " + paid;
    }
}

public class HospitalManagementSystem {
    private static Map<String, Patient> patients = new HashMap<>();
    private static Map<String, Staff> staff = new HashMap<>();
    private static Map<String, Appointment> appointments = new HashMap<>();
    private static Map<String, Bill> bills = new HashMap<>();
    private static Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        while (true) {
            System.out.println("\nHospital Management System");
            System.out.println("1. Patient Management");
            System.out.println("2. Staff Management");
            System.out.println("3. Appointments");
            System.out.println("4. Billing");
            System.out.println("5. Exit");
            System.out.print("Enter your choice: ");
            int choice = scanner.nextInt();
            scanner.nextLine(); // Consume newline

            switch (choice) {
                case 1 -> managePatients();
                case 2 -> manageStaff();
                case 3 -> manageAppointments();
                case 4 -> manageBilling();
                case 5 -> {
                    System.out.println("Exiting...");
                    return;
                }
                default -> System.out.println("Invalid choice. Please try again.");
            }
        }
    }

    private static void managePatients() {
        System.out.println("\nPatient Management");
        System.out.println("1. Add Patient");
        System.out.println("2. Search Patient");
        System.out.println("3. Update Patient Information");
        System.out.println("4. View All Patients");
        System.out.print("Enter your choice: ");
        int choice = scanner.nextInt();
        scanner.nextLine();

        switch (choice) {
            case 1 -> addPatient();
            case 2 -> searchPatient();
            case 3 -> updatePatient();
            case 4 -> viewPatients();
            default -> System.out.println("Invalid choice.");
        }
    }

    private static void addPatient() {
        System.out.print("Enter Name: ");
        String name = scanner.nextLine();
        System.out.print("Enter Age: ");
        int age = scanner.nextInt();
        scanner.nextLine();
        System.out.print("Enter Gender: ");
        String gender = scanner.nextLine();
        System.out.print("Enter Contact Number: ");
        String contact = scanner.nextLine();
        System.out.print("Enter Address: ");
        String address = scanner.nextLine();
        System.out.print("Enter Medical History: ");
        String history = scanner.nextLine();
        System.out.print("Enter Admission Date: ");
        String admissionDate = scanner.nextLine();

        String patientID = UUID.randomUUID().toString();
        Patient patient = new Patient(name, age, gender, contact, address, history, admissionDate, patientID);
        patients.put(patientID, patient);
        System.out.println("Patient added successfully.");
    }

    private static void searchPatient() {
        System.out.print("Enter Patient ID or Name: ");
        String input = scanner.nextLine();

        if (patients.containsKey(input)) {
            System.out.println(patients.get(input));
            return;
        }

        for (Patient patient : patients.values()) {
            if (patient.getName().equalsIgnoreCase(input)) {
                System.out.println(patient);
                return;
            }
        }

        System.out.println("Patient not found.");
    }

    private static void updatePatient() {
        System.out.print("Enter Patient ID: ");
        String patientID = scanner.nextLine();

        if (!patients.containsKey(patientID)) {
            System.out.println("Patient not found.");
            return;
        }

        Patient patient = patients.get(patientID);
        System.out.println("Updating details for Patient: " + patient.getName());

        System.out.print("Enter new Address (leave blank to keep unchanged): ");
        String address = scanner.nextLine();
        if (!address.isBlank()) patient.setAddress(address);

        System.out.print("Enter new Contact Number (leave blank to keep unchanged): ");
        String contact = scanner.nextLine();
        if (!contact.isBlank()) patient.setContactNumber(contact);

        System.out.print("Enter new Medical History (leave blank to keep unchanged): ");
        String history = scanner.nextLine();
        if (!history.isBlank()) patient.setMedicalHistory(history);

        System.out.println("Patient information updated successfully.");
    }

    private static void viewPatients() {
        if (patients.isEmpty()) {
            System.out.println("No patients found.");
            return;
        }

        System.out.println("\nList of All Patients:");
        for (Patient patient : patients.values()) {
            System.out.println(patient);
        }
    }

    private static void manageStaff() {
        System.out.println("\nStaff Management");
        System.out.println("1. Add Staff");
        System.out.println("2. Search Staff");
        System.out.println("3. Update Staff Information");
        System.out.print("Enter your choice: ");
        int choice = scanner.nextInt();
        scanner.nextLine();

        switch (choice) {
            case 1 -> addStaff();
            case 2 -> searchStaff();
            case 3 -> updateStaff();
            default -> System.out.println("Invalid choice.");
        }
    }

    private static void addStaff() {
        System.out.print("Enter Name: ");
        String name = scanner.nextLine();
        System.out.print("Enter Age: ");
        int age = scanner.nextInt();
        scanner.nextLine();
        System.out.print("Enter Gender: ");
        String gender = scanner.nextLine();
        System.out.print("Enter Contact: ");
        String contact = scanner.nextLine();
        System.out.print("Enter Role: ");
        String role = scanner.nextLine();
        System.out.print("Enter Department: ");
        String department = scanner.nextLine();

        String staffID = UUID.randomUUID().toString();
        Staff newStaff = new Staff(name, age, gender, contact, role, department, staffID);
        staff.put(staffID, newStaff);
        System.out.println("Staff added successfully.");
    }

    private static void searchStaff() {
        System.out.print("Enter Staff ID: ");
        String id = scanner.nextLine();
        if (staff.containsKey(id)) {
            System.out.println(staff.get(id));
        } else {
            System.out.println("Staff not found.");
        }
    }

    private static void updateStaff() {
        System.out.print("Enter Staff ID: ");
        String staffID = scanner.nextLine();

        if (!staff.containsKey(staffID)) {
            System.out.println("Staff not found.");
            return;
        }

        Staff staffMember = staff.get(staffID);
        System.out.println("Updating details for Staff: " + staffMember.getName());

        System.out.print("Enter new Role (leave blank to keep unchanged): ");
        String role = scanner.nextLine();
        if (!role.isBlank()) staffMember.setRole(role);

        System.out.print("Enter new Department (leave blank to keep unchanged): ");
        String department = scanner.nextLine();
        if (!department.isBlank()) staffMember.setDepartment(department);

        System.out.print("Enter new Contact Number (leave blank to keep unchanged): ");
        String contact = scanner.nextLine();
        if (!contact.isBlank()) staffMember.setContactNumber(contact);

        System.out.println("Staff information updated successfully.");
    }

    private static void manageAppointments() {
        System.out.println("\nAppointments Management");
        System.out.println("1. Schedule Appointment");
        System.out.println("2. View Appointments");
        System.out.print("Enter your choice: ");
        int choice = scanner.nextInt();
        scanner.nextLine();

        switch (choice) {
            case 1 -> scheduleAppointment();
            case 2 -> viewAppointments();
            default -> System.out.println("Invalid choice.");
        }
    }

    private static void scheduleAppointment() {
        System.out.print("Enter Patient ID: ");
        String patientID = scanner.nextLine();
        System.out.print("Enter Doctor ID: ");
        String doctorID = scanner.nextLine();
        System.out.print("Enter Date/Time: ");
        String dateTime = scanner.nextLine();
        System.out.print("Enter Reason: ");
        String reason = scanner.nextLine();

        Appointment appointment = new Appointment(patientID, doctorID, dateTime, reason);
        appointments.put(UUID.randomUUID().toString(), appointment);
        System.out.println("Appointment scheduled successfully.");
    }

    private static void viewAppointments() {
        if (appointments.isEmpty()) {
            System.out.println("No appointments found.");
            return;
        }
        for (Appointment appointment : appointments.values()) {
            System.out.println(appointment);
        }
    }

    private static void manageBilling() {
        System.out.println("\nBilling Management");
        System.out.println("1. Generate Bill");
        System.out.println("2. View Bill");
        System.out.print("Enter your choice: ");
        int choice = scanner.nextInt();
        scanner.nextLine();

        switch (choice) {
            case 1 -> generateBill();
            case 2 -> viewBill();
            default -> System.out.println("Invalid choice.");
        }
    }

    private static void generateBill() {
        System.out.print("Enter Patient ID: ");
        String patientID = scanner.nextLine();
        System.out.print("Enter Consultation Fee: ");
        double consultation = scanner.nextDouble();
        System.out.print("Enter Room Charges: ");
        double room = scanner.nextDouble();
        System.out.print("Enter Other Services: ");
        double services = scanner.nextDouble();
        scanner.nextLine();

        Bill bill = new Bill(patientID, consultation, room, services);
        bills.put(patientID, bill);
        System.out.println("Bill generated for Patient ID: " + patientID);
    }

    private static void viewBill() {
        System.out.print("Enter Patient ID: ");
        String id = scanner.nextLine();
        if (bills.containsKey(id)) {
            System.out.println(bills.get(id));
        } else {
            System.out.println("No bill found for the patient.");
        }
    }
}



pragma solidity ^0.8.0;

contract DataSharingContract {
    event DataShared(
        string indexed senderName,
        string indexed receiverName,
        uint256 timestamp,
        string dataSetName
    );

    struct SharedData {
        string senderName;
        string receiverName;
        uint256 timestamp;
        string dataSetName;
    }

    SharedData[] public sharedData;

    function shareData(
        string memory _senderName,
        string memory _receiverName,
        string memory _dataSetName
    ) public {
        SharedData memory data = SharedData(
            _senderName,
            _receiverName,
            block.timestamp,
            _dataSetName
        );

        sharedData.push(data);

        emit DataShared(_senderName, _receiverName, block.timestamp, _dataSetName);
    }
}
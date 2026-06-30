package entities;

import org.eclipse.mosaic.lib.objects.ToDataOutput;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

// Custom payload wrapper used to carry CPM JSON in a GenericV2xMessage.
public class CpmPayload implements ToDataOutput {
    private final String json;

    public CpmPayload(String json) {
        this.json = json;
    }

    public CpmPayload(DataInput dataInput) throws IOException {
        int length = dataInput.readInt();
        byte[] bytes = new byte[length];
        dataInput.readFully(bytes);
        this.json = new String(bytes, StandardCharsets.UTF_8);
    }

    public CpmPayload(DataInputStream dataInputStream) throws IOException {
        this((DataInput) dataInputStream);
    }

    public String getJson() {
        return json;
    }

    @Override
    public void toDataOutput(DataOutput dataOutput) throws IOException {
        // Prefix the UTF-8 JSON with its byte length for decoding on receive.
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        dataOutput.writeInt(bytes.length);
        dataOutput.write(bytes);
    }
}
